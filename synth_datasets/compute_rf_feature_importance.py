import numpy as np
from sklearn.ensemble._forest import (
    _generate_sample_indices,
    _generate_unsampled_indices,
)


def compute_rf_feature_importance(rf, X, y, loss, methods, X_test=None, y_test=None):
    """
    Compute the j-score, UFI or MDI-oob feature importance of a given random forest.
    """

    feature_importance = {method: np.zeros(X.shape[1]) for method in methods}
    for tree in rf.estimators_:
        fi_tree = {method: np.zeros(X.shape[1]) for method in methods}
        n_nodes = tree.tree_.node_count

        inb_indices = _generate_sample_indices(
            tree.random_state, rf._n_samples, rf._n_samples_bootstrap
        )
        X_inb = X[inb_indices]
        y_inb = y[inb_indices]
        decision_path_inb = np.array(tree.decision_path(X_inb).todense())
        omega_inb = np.sum(decision_path_inb, axis=0) / X_inb.shape[0]

        oob_indices = _generate_unsampled_indices(
            tree.random_state, rf._n_samples, rf._n_samples_bootstrap
        )
        X_oob = X[oob_indices]
        y_oob = y[oob_indices]
        decision_path_oob = np.array(tree.decision_path(X_oob).todense())
        omega_oob = np.sum(decision_path_oob, axis=0) / X_oob.shape[0]

        impurity = {method: np.zeros(n_nodes) for method in methods}
        has_oob_samples_in_children = [True] * n_nodes

        if X_test is not None and y_test is not None and "j-score_test" in methods:
            decision_path_test = np.array(tree.decision_path(X_test).todense())
            omega_test = np.sum(decision_path_test, axis=0) / X_test.shape[0]

            impurity = {method: np.zeros(n_nodes) for method in methods}
            has_test_samples_in_children = [True] * n_nodes

            do_test = True
        else:
            do_test = False

        for node_idx in range(n_nodes):  # Compute the "cross-impurity" of each node
            y_innode_oob = y_oob[np.where(decision_path_oob[:, node_idx].ravel())]
            y_innode_inb = y_inb[np.where(decision_path_inb[:, node_idx].ravel())]
            if do_test:
                y_innode_test = y_test[
                    np.where(decision_path_test[:, node_idx].ravel())
                ]

            if len(y_innode_oob) == 0:  # If no oob in node, skip and flag the node
                if sum(tree.tree_.children_left == node_idx) > 0:
                    parent_node = np.arange(n_nodes)[
                        tree.tree_.children_left == node_idx
                    ][0]
                    has_oob_samples_in_children[parent_node] = False
                else:
                    parent_node = np.arange(n_nodes)[
                        tree.tree_.children_right == node_idx
                    ][0]
                    has_oob_samples_in_children[parent_node] = False
            else:
                if loss in ["gini", "brier"]:
                    p_node_oob = np.unique(y_innode_oob, return_counts=True)[1] / len(
                        y_innode_oob
                    )
                    p_node_inb = np.unique(y_innode_inb, return_counts=True)[1] / len(
                        y_innode_inb
                    )
                    if "j-score" in methods:
                        impurity["j-score"][node_idx] = np.sum(
                            p_node_oob - 2 * p_node_oob * p_node_inb + p_node_inb**2
                        )
                    if "UFI" in methods:
                        impurity["UFI"][node_idx] = np.sum(1 - p_node_oob * p_node_inb)
                    if "MDI-oob" in methods:
                        impurity["MDI-oob"][node_idx] = np.sum(
                            1 - p_node_oob * p_node_inb
                        )

                elif loss in ["mse", "squared_error"]:
                    if "j-score" in methods:
                        impurity["j-score"][node_idx] = (
                            np.mean((y_innode_oob - y_innode_inb.mean()) ** 2)
                        )
                    if "UFI" in methods:
                        impurity["UFI"][node_idx] = (
                            np.mean((y_innode_oob - y_innode_inb.mean()) ** 2)
                            + np.mean((y_innode_inb - y_innode_inb.mean()) ** 2)
                        ) / 2
                    if "MDI-oob" in methods:
                        impurity["MDI-oob"][node_idx] = (
                            -y_innode_oob.mean() * y_innode_inb.mean()
                        )
                if do_test:
                    if (
                        len(y_innode_test) == 0
                    ):  # If no test in node, skip and flag the node
                        if sum(tree.tree_.children_left == node_idx) > 0:
                            parent_node = np.arange(n_nodes)[
                                tree.tree_.children_left == node_idx
                            ][0]
                            has_test_samples_in_children[parent_node] = False
                        else:
                            parent_node = np.arange(n_nodes)[
                                tree.tree_.children_right == node_idx
                            ][0]
                            has_test_samples_in_children[parent_node] = False
                    else:
                        if loss in ["gini", "brier"]:
                            p_node_test = np.unique(y_innode_test, return_counts=True)[
                                1
                            ] / len(y_innode_test)
                            p_node_inb = np.unique(y_innode_inb, return_counts=True)[
                                1
                            ] / len(y_innode_inb)
                            impurity["j-score_test"][node_idx] = np.sum(
                                p_node_test
                                - 2 * p_node_test * p_node_inb
                                + p_node_inb**2
                            )
                        elif loss in ["mse", "squared_error"]:
                            impurity["j-score_test"][node_idx] = (
                                np.mean(y_innode_test**2)
                                - 2 * y_innode_test.mean() * y_innode_inb.mean()
                                + y_innode_inb.mean() ** 2
                            )
        for node_idx in range(n_nodes):
            if (
                tree.tree_.children_left[node_idx] == -1
                or tree.tree_.children_right[node_idx] == -1
            ):
                continue

            feature_idx = tree.tree_.feature[node_idx]

            node_left = tree.tree_.children_left[node_idx]
            node_right = tree.tree_.children_right[node_idx]

            if has_oob_samples_in_children[node_idx]:
                if "j-score" in methods:
                    fi_tree["j-score"][feature_idx] += (
                        omega_oob[node_idx] * impurity["j-score"][node_idx]
                        - omega_oob[node_left] * impurity["j-score"][node_left]
                        - omega_oob[node_right] * impurity["j-score"][node_right]
                    )
                if "MDI-oob" in methods:
                    fi_tree["MDI-oob"][feature_idx] += (
                        omega_oob[node_idx] * impurity["MDI-oob"][node_idx]
                        - omega_oob[node_left] * impurity["MDI-oob"][node_left]
                        - omega_oob[node_right] * impurity["MDI-oob"][node_right]
                    )
                if "UFI" in methods:
                    fi_tree["UFI"][feature_idx] += (
                        omega_inb[node_idx] * impurity["UFI"][node_idx]
                        - omega_inb[node_left] * impurity["UFI"][node_left]
                        - omega_inb[node_right] * impurity["UFI"][node_right]
                    )
            if do_test and has_test_samples_in_children[node_idx]:
                fi_tree["j-score_test"][feature_idx] += (
                    omega_test[node_idx] * impurity["j-score_test"][node_idx]
                    - omega_test[node_left] * impurity["j-score_test"][node_left]
                    - omega_test[node_right] * impurity["j-score_test"][node_right]
                )
        for method in methods:
            feature_importance[method] += fi_tree[method]
    for method in methods:
        feature_importance[method] /= rf.n_estimators

    return feature_importance
