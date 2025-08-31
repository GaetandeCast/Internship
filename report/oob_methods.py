import numpy as np
from sklearn.ensemble._forest import (
    _generate_sample_indices,
    _generate_unsampled_indices,
)


def compute_oob_fis(rf, X, y, loss, methods):
    """
    Compute the j-score, UFI, MDI-oob, or naive-oob feature importance of a given random forest.
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

        impurity = np.zeros(n_nodes)
        cross_impurity = np.zeros(n_nodes)
        oob_impurity = np.zeros(n_nodes)
        has_oob_samples_in_children = [True] * n_nodes


        for node_idx in range(n_nodes):  # Compute the "cross-impurity" of each node
            y_innode_oob = y_oob[np.where(decision_path_oob[:, node_idx].ravel())]
            y_innode_inb = y_inb[np.where(decision_path_inb[:, node_idx].ravel())]

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
                if "UFI" in methods or "MDI-oob" in methods:
                    impurity[node_idx] = loss(y_innode_inb, np.repeat(y_innode_inb.mean(), len(y_innode_inb)))
                if "naive-oob" in methods:
                    oob_impurity[node_idx] = loss(y_innode_oob, np.repeat(y_innode_oob.mean(), len(y_innode_oob)))
                cross_impurity[node_idx] = loss(y_innode_oob, np.repeat(y_innode_inb.mean(), len(y_innode_oob)))

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
                        omega_oob[node_idx] * cross_impurity[node_idx]
                        - omega_oob[node_left] * cross_impurity[node_left]
                        - omega_oob[node_right] * cross_impurity[node_right]
                    )
                if "MDI-oob" in methods:
                    fi_tree["MDI-oob"][feature_idx] += (
                        omega_oob[node_idx] * (impurity[node_idx] + cross_impurity[node_idx]) / 2
                        - omega_oob[node_left] * (impurity[node_left] + cross_impurity[node_left]) / 2
                        - omega_oob[node_right] * (impurity[node_right] + cross_impurity[node_right]) / 2
                    )
                if "UFI" in methods:
                    fi_tree["UFI"][feature_idx] += (
                        omega_inb[node_idx] * (impurity[node_idx] + cross_impurity[node_idx]) / 2
                        - omega_inb[node_left] * (impurity[node_left] + cross_impurity[node_left]) / 2
                        - omega_inb[node_right] * (impurity[node_right] + cross_impurity[node_right]) / 2
                    )                
                if "naive-oob" in methods:
                    fi_tree["naive-oob"][feature_idx] += (
                        omega_inb[node_idx] * oob_impurity[node_idx]
                        - omega_inb[node_left] * oob_impurity[node_left]
                        - omega_inb[node_right] * oob_impurity[node_right]
                    )
        for method in methods:
            feature_importance[method] += fi_tree[method]
    for method in methods:
        feature_importance[method] /= rf.n_estimators

    return feature_importance
