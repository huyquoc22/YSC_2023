import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

if __name__ == "__main__":
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print("#training images = %d" % dataset_size)

    model = create_model(opt)
    model.initialize(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    # Grid Search
    opt_list = []

    # Tạo các tùy chọn khác nhau
    gpu_ids_options = [[0], [0, 1]]  # Tùy chọn cho 'gpu_ids'
    niter_options = [10000, 20000]  # Tùy chọn cho 'niter'
    lr_options = [0.0002, 0.0005]  # Tùy chọn cho 'lr'

# Tạo tất cả các kết hợp của các tùy chọn
    for gpu_ids in gpu_ids_options:
        for niter in niter_options:
            for lr in lr_options:
                opt = TrainOptions().parse()
                opt.gpu_ids = gpu_ids
                opt.niter = niter
                opt.lr = lr
                opt_list.append(opt)

    best_opt = model.grid_search(opt_list)
    print("Best option:", best_opt)

    # Tiếp tục huấn luyện với tùy chọn tốt nhất
    opt = best_opt
    model.setup(opt)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()
            # print('1')

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, save_result
                )

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(
                    epoch, epoch_iter, losses, t, iter_start_time - iter_data_time
                )
                if opt.display_id > 0:
                    visualizer.plot_current_losses(
                        epoch, float(epoch_iter) / dataset_size, opt, losses
                    )

            if total_steps % opt.save_latest_freq == 0:
                print(
                    "saving the latest model (epoch %d, total_steps %d)"
                    % (epoch, total_steps)
                )
                model.save_networks("latest")

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print(
                "saving the model at the end of epoch %d, iters %d"
                % (epoch, total_steps)
            )
            model.save_networks("latest")
            model.save_networks(epoch)

        model.update_learning_rate()
