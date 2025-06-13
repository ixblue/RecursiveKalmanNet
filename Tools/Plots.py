import matplotlib.pyplot as plt
import torch

color_oKF = '#1E88E5'
color_soKF = '#D81B60'
color_CKN = '#004D40'
color_RKN = '#E65100'
color_sup = '#6A1B9A'

marker_every = 10
marker_size = 3
marker_oKF = 'd'
marker_soKF = 'v'
marker_CKN = '^'
marker_RKN = 'x'

colors_list = ['#1E88E5', '#D81B60', '#004D40', '#E65100', '#6A1B9A', '#FFB300']
markers_list = ['d', 'v', '^', 'x', '*']


def Plot_Comparison_RKN(T, path, data_dict, y_lim=None, show=False, marker_every=None, graph_size=1, time_lim=None, x_label="Time", y_label='Position Std', aspect_ratio=[3.5,2.5], fontsize=8, title=None, linewidth=1, marker_size=3):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["text.usetex"] = True 
    plt.rcParams['font.size'] = fontsize
    
    row_number, col_number = next(iter(data_dict.values())).size()[0], next(iter(data_dict.values())).size()[1]
    
    fig, axs = plt.subplots(row_number, col_number, figsize=(col_number * aspect_ratio[0] * graph_size, row_number * aspect_ratio[1] * graph_size))
    
    for row in range(row_number):
        for col in range(col_number):
            for i, (key, data) in enumerate(data_dict.items()):
                color = colors_list[i % len(colors_list)]
                marker = markers_list[i % len(markers_list)]
                
                plot_target = data[row, col, time_lim[0] if time_lim else 0: time_lim[1] if time_lim else T].detach().numpy()
                
                if col_number == 1 and row_number > 1:
                    axs[row].plot(plot_target, label=key, color=color, linewidth=linewidth, marker=marker, markevery=marker_every, markersize=marker_size)
                    axs[row].legend(loc='upper right',fontsize=fontsize-1)
                    axs[row].grid(linewidth=0.7, color='lightgray', alpha=0.7)
                    axs[row].set_xlabel(x_label, fontsize=fontsize)
                    axs[row].set_ylabel(y_label, fontsize=fontsize)
                    if y_lim is not None:
                        axs[row].set_ylim(y_lim[row])
                elif row_number == 1:
                    axs.plot(plot_target, label=key, color=color, linewidth=linewidth, marker=marker, markevery=marker_every, markersize=marker_size)
                    axs.legend(loc='upper right',fontsize=fontsize-1)
                    axs.grid(linewidth=0.7, color='lightgray', alpha=0.7)
                    axs.set_xlabel(x_label, fontsize=fontsize)
                    axs.set_ylabel(y_label, fontsize=fontsize)
                    if y_lim is not None:
                        axs.set_ylim(y_lim[row])
                else:
                    axs[row, col].plot(plot_target, label=key, color=color, linewidth=linewidth, marker=marker, markevery=marker_every, markersize=marker_size)
                    axs[row,col].legend(loc='upper right',fontsize=fontsize-1)
                    axs[row,col].grid(linewidth=0.7, color='lightgray', alpha=0.7)
                    axs[row,col].set_xlabel(x_label, fontsize=fontsize)
                    axs[row,col].set_ylabel(y_label, fontsize=fontsize)
                    if y_lim is not None:
                        axs[row, col].set_ylim(y_lim[row])
    if title:
        plt.title(title)
    if path:
        plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0)
    if show:
        fig.tight_layout()



def Plot_Std_RKN(T, state, path, test_target, data_dict, y_lim=None, marker_every=None, graph_size=1, time_lim=None,\
                      x_label="Time", y_label='Position Std', err_emp_std="Emp. std", fontsize=8, aspect_ratio=[3.5,2.5], title=None, linewidth=1):
    linestyle_filter_std = '-.'
    linestyle_error_std = '-'

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["text.usetex"] = True 
    plt.rcParams['font.size'] = fontsize

    batch_size, state_dim, T = test_target.size()

    t_min = time_lim[0] if time_lim else 0
    t_max = time_lim[1] if time_lim else T

    # Calcul des erreurs et des écarts-types
    errors = {key: torch.zeros([batch_size, 1, T]) for key in data_dict}
    std_devs = {key: torch.zeros([1, T]) for key in data_dict}
    for t in range(T):
        for key in data_dict:
            errors[key][:, 0, t] = test_target[:, state, t] - data_dict[key][0][:, state, t]
            std_devs[key][0, t] = torch.sqrt(torch.mean(torch.square(errors[key][:, 0, t] - torch.mean(errors[key][:, 0, t]))))
    
    # Affichage des courbes
    plt.figure(figsize=(graph_size*aspect_ratio[0], graph_size*aspect_ratio[1]))
    for i, key in enumerate(data_dict.keys()):
        index = data_dict[key][2]
        color = colors_list[index % len(colors_list)]
        marker = markers_list[index % len(markers_list)]
        
        plt.plot(std_devs[key][0, t_min:t_max].detach().numpy(), label=f'{key} : '+err_emp_std,
                 color=color, linestyle=linestyle_error_std, linewidth=linewidth,
                 marker=marker, markevery=marker_every, markersize=marker_size)
        
    for i, key in enumerate(data_dict.keys()):
        index = data_dict[key][2]
        color = colors_list[index % len(colors_list)]
        marker = markers_list[index % len(markers_list)]
        
        plt.plot(torch.sqrt(torch.mean(data_dict[key][1][:, state, state], dim=0))[t_min:t_max].detach().numpy(),
                 label=f'{key} : P', color=color, linestyle=linestyle_filter_std,
                 linewidth=linewidth, marker=marker, markevery=marker_every, markersize=marker_size)
        
    if y_lim!=None:
        plt.ylim(y_lim)
    plt.xlabel(x_label, fontsize=fontsize)
    plt.ylabel(y_label, fontsize=fontsize)
    plt.grid(linewidth=0.7, color='lightgray', alpha=0.7)
    plt.legend(loc='upper right', fontsize=fontsize-1, labelspacing=0.10, ncol=2, columnspacing=0.2, handleheight=2, handletextpad=1)
    if title:
        plt.title(title)
    if path:
        plt.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0)


def Plot_Loss(model_name, aspect_ratio=[3.5, 3], fontsize=12):
    """
    Plot the loss curves for a given model by giving the model name.
    The data can be loaded from the file '.results/loss_curves/{model_name}.pt' if the user wants to create a specific plot.
    """
    losses = torch.load(f'.results/loss_curves/{model_name}.pt')

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["text.usetex"] = True
    plt.rcParams['font.size'] = fontsize

    plt.figure(figsize=aspect_ratio)
    plt.plot(losses['train_losses'].detach().numpy(), label = 'Training Curve')
    plt.plot(losses['cv_losses'].detach().numpy(), label = 'Validation Curve')
    plt.legend()
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')



def Plot_SMD(args, path, test_target, oKF_x, oKF_cov, soKF_x, soKF_cov, RKN_x, RKN_cov, CKN_x = None, CKN_cov = None, y_lim=None, marker_every=None):
    state = 0
    linestyle_error_std = '-'
    linewidth = 1    
    fontsize=8
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["text.usetex"] = True

    err_emp_std = "Emp. std"
    plt.rcParams['font.size'] = fontsize
    plt.figure(figsize=(3.5,3))

    # Calcul pour le graphe des std
    oKF_n_error = torch.zeros([args.N_T, args.T_test])
    soKF_n_error= torch.zeros([args.N_T, args.T_test])
    if CKN_x is not None:
        CKN_n_error = torch.zeros([args.N_T, args.T_test])
    RKN_n_error = torch.zeros([args.N_T, args.T_test])


    for t in range(args.T_test):
        oKF_n_error[:, t] = torch.bmm( torch.bmm(torch.sub(test_target[:, :, t], oKF_x[:, :, t]).unsqueeze(1), torch.linalg.inv(oKF_cov[:, :, :, t])), torch.sub(test_target[:, :, t], oKF_x[:, :, t]).unsqueeze(2)).squeeze()
        soKF_n_error[:, t] = torch.bmm( torch.bmm(torch.sub(test_target[:, :, t], soKF_x[:, :, t]).unsqueeze(1), torch.linalg.inv(soKF_cov[:, :, :, t])), torch.sub(test_target[:, :, t], soKF_x[:, :, t]).unsqueeze(2)).squeeze()
        if CKN_x is not None:
            CKN_n_error[:, t] = torch.bmm( torch.bmm(torch.sub(test_target[:, :, t], CKN_x[:, :, t+1]).unsqueeze(1), torch.linalg.inv(CKN_cov[:, :, :, t])), torch.sub(test_target[:, :, t], CKN_x[:, :, t+1]).unsqueeze(2)).squeeze()
        RKN_n_error[:, t] = torch.bmm( torch.bmm(torch.sub(test_target[:, :, t], RKN_x[:, :, t]).unsqueeze(1), torch.linalg.inv(RKN_cov[:, :, :, t])), torch.sub(test_target[:, :, t], RKN_x[:, :, t]).unsqueeze(2)).squeeze()

    RMS_oKF = torch.zeros([args.T_test])
    RMS_soKF = torch.zeros([args.T_test])
    if CKN_x is not None:
        RMS_CKN = torch.zeros([args.T_test])
    RMS_RKN = torch.zeros([args.T_test])

    for t in range(args.T_test):
        RMS_oKF[t] = torch.mean(oKF_n_error[:, t], dim=0)
        RMS_soKF[t] = torch.mean(soKF_n_error[:, t], dim=0)
        if CKN_x is not None:
            RMS_CKN[t] = torch.mean(CKN_n_error[:, t], dim=0)
        RMS_RKN[t] = torch.mean(RKN_n_error[:, t], dim=0)

    plt.plot(RMS_oKF.detach().numpy(), label = 'o-KF', color=color_oKF, linestyle=linestyle_error_std, linewidth=linewidth, marker=marker_oKF, markevery=marker_every, markersize=marker_size)
    plt.plot(RMS_soKF.detach().numpy(), label = 'so-KF', color=color_soKF, linestyle=linestyle_error_std, linewidth=linewidth, marker=marker_soKF, markevery=marker_every, markersize=marker_size)
    if CKN_x is not None:
        plt.plot(RMS_CKN.detach().numpy(), label = 'CKN', color=color_CKN, linestyle=linestyle_error_std, linewidth=linewidth, marker=marker_CKN, markevery=marker_every, markersize=marker_size)
    plt.plot(RMS_RKN.detach().numpy(), label = 'RKN', color=color_RKN, linestyle=linestyle_error_std, linewidth=linewidth, marker=marker_RKN, markevery=marker_every, markersize=marker_size)

    plt.xlabel('Time', fontsize=fontsize)
    plt.ylabel('MSMD', fontsize=fontsize)
    plt.grid(linewidth=0.7, color='lightgray', alpha=0.7)
    plt.legend(loc='upper right', fontsize=fontsize-1, labelspacing=0.15,ncol=2, columnspacing=0.2)
    if path:
        plt.savefig(path, format="pdf", bbox_inches="tight")

    print("Mean value of the SMD over batch and time")
    print("o-KF : ",torch.mean(oKF_n_error[:,:]))
    print("so-KF : ",torch.mean(soKF_n_error[:,:]))
    if CKN_x is not None:
        print("CKN : ",torch.mean(CKN_n_error[:,:]))
    print("RKN : ",torch.mean(RKN_n_error[:,:]))


