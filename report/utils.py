import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats



def check_markov_blanket_found(importance_scores, markov_blanket_features=[1, 2, 4, 7]):
    """
    Check if the top-ranked features match the Markov blanket.
    
    Parameters:
    -----------
    importance_scores : array-like
        Feature importance scores for features [0,1,2,3,4,6,7] (feature 5 is target, skipped)
    markov_blanket_features : list
        Original feature indices that form the Markov blanket
    
    Returns:
    --------
    bool : True if Markov blanket was found, False otherwise
    """
    # Map original feature indices to importance array indices
    # Original features: [0,1,2,3,4,5,6,7] -> [0,1,2,3,4,target,6,7]
    # Importance array: [0,1,2,3,4,5,6] corresponds to features [0,1,2,3,4,6,7]
    def original_to_importance_index(original_idx):
        if original_idx < 5:
            return original_idx
        elif original_idx > 5:
            return original_idx - 1
        else:
            raise ValueError("Feature 5 is the target, no importance score available")
    
    # Convert Markov blanket features to importance array indices
    mb_importance_indices = [original_to_importance_index(f) for f in markov_blanket_features]
    
    # Get indices of features sorted by importance (descending order)
    top_features = np.argsort(importance_scores)[::-1]
    
    # Get the top N features where N is the size of the Markov blanket
    top_n_features = set(top_features[:len(mb_importance_indices)])
    markov_set = set(mb_importance_indices)
    
    # Check if top features match the Markov blanket
    return top_n_features == markov_set

def compute_markov_blanket_detection_rates(results_dict, method_name, markov_blanket_features=[1, 2, 4, 7]):
    """
    Compute how many times a method found the Markov blanket.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with simulation results
    method_name : str
        Name of the method for display
    markov_blanket_features : list
        Indices of features that form the Markov blanket (0-indexed)
    
    Returns:
    --------
    tuple : (success_count, total_simulations, success_rate)
    """
    success_count = 0
    total_simulations = len(results_dict)
    
    for sim in range(total_simulations):
        if check_markov_blanket_found(results_dict[sim], markov_blanket_features):
            success_count += 1
    
    success_rate = success_count / total_simulations if total_simulations > 0 else 0
    return success_count, total_simulations, success_rate

def prepare_boxplot_data(methods_dict, n_simulations):
    """
    Prepare data for boxplot visualization.
    
    Parameters:
    -----------
    methods_dict : dict
        Dictionary of method names and their results
    n_simulations : int
        Number of simulations
    
    Returns:
    --------
    pandas.DataFrame : Long format data for plotting
    """
    data_rows = []
    
    # Feature mapping: importance indices to original feature indices
    # Importance array: [0,1,2,3,4,5,6] corresponds to original features [0,1,2,3,4,6,7]
    importance_to_original = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 7, 7:8}
    
    for method_name, results_dict in methods_dict.items():
        for sim in range(n_simulations):
            if sim in results_dict:
                importance_scores = results_dict[sim]
                
                for imp_idx, score in enumerate(importance_scores):
                    original_feature = importance_to_original[imp_idx]
                    
                    data_rows.append({
                        'Method': method_name,
                        'Feature': f'Feature {original_feature}',
                        'Original_Feature_Index': original_feature,
                        'Importance': score,
                        'Simulation': sim,
                        'Is_Markov_Blanket': original_feature in [1, 2, 4, 7]
                    })
    
    return pd.DataFrame(data_rows)

def create_feature_importance_boxplots(methods_dict, n_simulations, figsize=(10, 20)):
    """
    Create comprehensive boxplot visualization.
    
    Parameters:
    -----------
    methods_dict : dict
        Dictionary of method names and their results
    n_simulations : int
        Number of simulations
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    matplotlib.figure.Figure : The created figure
    """
    # Prepare data
    df = prepare_boxplot_data(methods_dict, n_simulations)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots for each method
    n_methods = len(methods_dict)
    fig, axes = plt.subplots(n_methods, 1, figsize=figsize, sharex=True)
    
    # Handle case with single method
    if n_methods == 1:
        axes = [axes]
    
    # Define colors for Markov blanket vs non-Markov blanket features
    markov_features = [1, 2, 4, 7]
    feature_colors = {}
    for feature in [0, 1, 2, 3, 4, 6, 7, 8]:
        if feature in markov_features:
            feature_colors[f'Feature {feature}'] = 'lightcoral'  # Red for Markov blanket
        else:
            feature_colors[f'Feature {feature}'] = 'lightblue'   # Blue for non-Markov blanket
    
    # Create boxplot for each method
    for i, method_name in enumerate(methods_dict.keys()):
        method_data = df[df['Method'] == method_name]
        
        # Create boxplot
        box_plot = axes[i].boxplot(
            [method_data[method_data['Original_Feature_Index'] == feat]['Importance'].values 
             for feat in [0, 1, 2, 3, 4, 6, 7, 8]],
            tick_labels=[f'F{feat}' for feat in [0, 1, 2, 3, 4, 6, 7, 8]],
            patch_artist=True,
            notch=True,
            showmeans=True,
            meanline=True
        )
        
        # Color the boxes
        for patch, feature in zip(box_plot['boxes'], [0, 1, 2, 3, 4, 6, 7, 8]):
            color = 'lightcoral' if feature in markov_features else 'lightblue'
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize the subplot
        axes[i].set_title(f'{method_name} - Feature Importance Distribution', 
                         fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Importance Score', fontsize=10)
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at zero
        axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Rotate x-axis labels if needed
        axes[i].tick_params(axis='x', rotation=0)
    
    # Set common x-label
    axes[-1].set_xlabel('Features (F5 is target, excluded)', fontsize=12, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightcoral', alpha=0.7, label='Markov Blanket (1,2,4,7)'),
        Patch(facecolor='lightblue', alpha=0.7, label='Non-Markov Blanket (0,3,6)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for legend
    
    return fig

def create_comparative_boxplot(methods_dict, n_simulations, figsize=(20, 10)):
    """
    Create a single plot comparing all methods side by side.
    
    Parameters:
    -----------
    methods_dict : dict
        Dictionary of method names and their results
    n_simulations : int
        Number of simulations
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    matplotlib.figure.Figure : The created figure
    """
    # Prepare data
    df = prepare_boxplot_data(methods_dict, n_simulations)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Create the boxplot using seaborn for better handling of multiple categories
    sns.boxplot(data=df, x='Feature', y='Importance', hue='Method', ax=ax, 
                notch=True, showmeans=True)
    
    # Customize the plot
    ax.set_title('Feature Importance Comparison Across Methods', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Features (F5 is target, excluded)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45)
    
    # Highlight Markov blanket features
    markov_positions = [1, 2, 4, 6]  # Positions of features 1,2,4,7 in the plot (0-indexed)
    for pos in markov_positions:
        ax.axvspan(pos-0.4, pos+0.4, alpha=0.1, color='red', zorder=0)
    
    # Add text annotation for Markov blanket
    ax.text(0.02, 0.98, 'Red shaded: Markov Blanket features (1,2,4,7)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust legend
    ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    return fig

def print_summary_statistics(methods_dict, n_simulations):
    """
    Print summary statistics for each method and feature.
    """
    df = prepare_boxplot_data(methods_dict, n_simulations)
    
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print("Mean importance scores by method and feature:")
    print("-" * 50)
    
    # Calculate means for each method-feature combination
    summary = df.groupby(['Method', 'Original_Feature_Index'])['Importance'].agg([
        'mean', 'std', 'median', 'min', 'max'
    ]).round(4)
    
    for method in methods_dict.keys():
        print(f"\n{method}:")
        method_summary = summary.loc[method]
        
        print("  Feature | Mean     | Std      | Median   | Min      | Max      | MB?")
        print("  --------|----------|----------|----------|----------|----------|----")
        
        for feature in [0, 1, 2, 3, 4, 6, 7, 8]:
            if feature in method_summary.index:
                stats = method_summary.loc[feature]
                is_mb = "Yes" if feature in [1, 2, 4, 7] else "No"
                print(f"  {feature:7d} | {stats['mean']:8.4f} | {stats['std']:8.4f} | "
                      f"{stats['median']:8.4f} | {stats['min']:8.4f} | {stats['max']:8.4f} | {is_mb:3s}")

def test_zero_importance(importance_scores, feature_idx=8, alpha=0.05):
    """
    Test if a feature has statistically zero importance using one-sample t-test.
    
    Parameters:
    -----------
    importance_scores : array-like
        Array of importance scores across simulations for the feature
    feature_idx : int
        Index of the feature being tested (for display)
    alpha : float
        Significance level for the test
    
    Returns:
    --------
    dict : Test results including statistic, p-value, and interpretation
    """
    scores = np.array(importance_scores)
    
    # Remove any NaN or infinite values
    scores = scores[np.isfinite(scores)]
    
    if len(scores) == 0:
        return {
            'feature_idx': feature_idx,
            'n_samples': 0,
            'mean': np.nan,
            'std': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'is_zero': False,
            'method': 'insufficient_data'
        }
    
    # One-sample t-test against null hypothesis: mean = 0
    t_stat, p_value = stats.ttest_1samp(scores, 0.0)
    
    # Alternative: Wilcoxon signed-rank test (non-parametric)
    try:
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(scores, alternative='two-sided')
    except ValueError:
        # All values might be zero
        wilcoxon_stat, wilcoxon_p = np.nan, 1.0
    
    return {
        'feature_idx': feature_idx,
        'n_samples': len(scores),
        'mean': np.mean(scores),
        'std': np.std(scores, ddof=1),
        'median': np.median(scores),
        't_statistic': t_stat,
        'p_value': p_value,
        'wilcoxon_statistic': wilcoxon_stat,
        'wilcoxon_p_value': wilcoxon_p,
        'is_zero_ttest': p_value > alpha,  # Fail to reject H0: mean = 0
        'is_zero_wilcoxon': wilcoxon_p > alpha,  # Fail to reject H0: median = 0
        'alpha': alpha
    }

def extract_random_feature_importance(results_dict, random_feature_original_idx=8):
    """
    Extract importance scores for the random feature across all simulations.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with simulation results
    random_feature_original_idx : int
        Original index of the random feature
    
    Returns:
    --------
    array : Importance scores for the random feature across simulations
    """
    # Convert original index to importance array index
    # Original features: [0,1,2,3,4,5,6,7,8] where 5 is target
    # Importance array: [0,1,2,3,4,5,6,7] corresponds to [0,1,2,3,4,6,7,8]
    if random_feature_original_idx < 5:
        importance_idx = random_feature_original_idx
    elif random_feature_original_idx > 5:
        importance_idx = random_feature_original_idx - 1
    else:
        raise ValueError("Feature 5 is the target")
    
    scores = []
    for sim in range(len(results_dict)):
        if sim in results_dict:
            scores.append(results_dict[sim][importance_idx])
    
    return np.array(scores)

def analyze_random_feature_importance(methods_dict, random_feature_idx=8, alpha=0.05):
    """
    Analyze which methods assign zero importance to the random feature.
    
    Parameters:
    -----------
    methods_dict : dict
        Dictionary of method names and their results
    random_feature_idx : int
        Original index of the random feature
    alpha : float
        Significance level
    
    Returns:
    --------
    dict : Results for each method
    """
    results = {}
    
    print(f"Statistical Test for Random Feature (Index {random_feature_idx})")
    print("=" * 70)
    print(f"H0: Feature {random_feature_idx} has zero importance (mean = 0)")
    print(f"H1: Feature {random_feature_idx} has non-zero importance (mean ≠ 0)")
    print(f"Significance level: α = {alpha}")
    print()
    
    for method_name, results_dict in methods_dict.items():
        # Extract importance scores for random feature
        scores = extract_random_feature_importance(results_dict, random_feature_idx)
        
        # Perform statistical test
        test_result = test_zero_importance(scores, random_feature_idx, alpha)
        results[method_name] = test_result
        
        # Display results
        print(f"{method_name:12} | n={test_result['n_samples']:2d} | "
              f"mean={test_result['mean']:8.4f} | std={test_result['std']:8.4f} | "
              f"t={test_result['t_statistic']:7.3f} | p={test_result['p_value']:7.4f} | "
              f"Zero: {'YES' if test_result['is_zero_ttest'] else 'NO':3s}")
    
    print("\n" + "=" * 70)
    
    # Summary
    zero_methods_ttest = [name for name, res in results.items() if res['is_zero_ttest']]
    zero_methods_wilcoxon = [name for name, res in results.items() if res['is_zero_wilcoxon']]
    
    print(f"Methods that assign ZERO importance (t-test, α={alpha}):")
    if zero_methods_ttest:
        print(f"  {', '.join(zero_methods_ttest)}")
    else:
        print("  None")
    
    print(f"\nMethods that assign ZERO importance (Wilcoxon test, α={alpha}):")
    if zero_methods_wilcoxon:
        print(f"  {', '.join(zero_methods_wilcoxon)}")
    else:
        print("  None")
    
    return results

def plot_random_feature_distributions(methods_dict, random_feature_idx=8):
    """
    Plot distributions of importance scores for the random feature.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (method_name, results_dict) in enumerate(methods_dict.items()):
        if i >= len(axes):
            break
            
        scores = extract_random_feature_importance(results_dict, random_feature_idx)
        
        # Histogram
        axes[i].hist(scores, bins=20, alpha=0.7, edgecolor='black')
        axes[i].axvline(0, color='red', linestyle='--', label='Zero importance')
        axes[i].axvline(np.mean(scores), color='green', linestyle='-', 
                       label=f'Mean = {np.mean(scores):.4f}')
        axes[i].set_title(f'{method_name}\nRandom Feature {random_feature_idx}')
        axes[i].set_xlabel('Importance Score')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(methods_dict), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return fig

