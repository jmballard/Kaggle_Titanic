import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_hist(
    df,
    var_to_plot,
    var_hue,
    save_fig = True,
    save_location = None, ):
    '''
    

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    var_to_plot : TYPE
        DESCRIPTION.
    var_hue : TYPE
        DESCRIPTION.
    save_fig : TYPE, optional
        DESCRIPTION. The default is True.
    save_location : TYPE, optional
        DESCRIPTION. The default is None.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    
    fig, ax = plt.subplots(figsize = (5,4))
    sns.histplot(
        x = df[var_to_plot][~df[var_hue].isna()],
        hue = df[var_hue][~df[var_hue].isna()],
        )
    ax.set_title(f"hist {var_to_plot} vs {var_hue}")
    plt.xticks(rotation = 'vertical')

    if save_fig == True:
        plt.savefig(
            os.path.join(
                save_location,
                f"hist {var_to_plot} vs {var_hue}.jpg"
                )
            )
    else:
        plt.show()