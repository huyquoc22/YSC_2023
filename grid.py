import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import itertools


def compute_score(opt):
    # Load the validation data using CreateDataLoader and load_data
    val_data_loader = CreateDataLoader(opt, is_train=False)
    val_dataset = val_data_loader.load_data()
    val_dataset_size = len(val_data_loader)
    print("#validation images = %d" % val_dataset_size)

    # Create and initialize the model for validation
    model = create_model(opt)
    model.setup(opt)

    total_loss = 0.0
    num_batches = 0

    # Validation loop
    for i, data in enumerate(val_dataset):
        model.set_input(data)
        model.test()  # Run the model in evaluation mode
        loss = model.get_current_losses()  # Replace with the actual loss function you use
        total_loss += loss.item()
        num_batches += 1

    # Compute the average loss
    average_loss = total_loss / num_batches

    # Return the average loss as the score
    return average_loss


def perform_grid_search(hyperparameter_grid):
    best_score = float('-inf')
    best_hyperparameters = None

    for hyperparameters in hyperparameter_grid:
        # Parse the hyperparameters
        opt = TrainOptions().parse(args=hyperparameters)

        # Load the data
        data_loader = CreateDataLoader(opt)
        dataset = data_loader.load_data()
        dataset_size = len(data_loader)
        print("#training images = %d" % dataset_size)

        # Create and initialize the model
        model = create_model(opt)
        model.setup(opt)
        visualizer = Visualizer(opt)
        total_steps = 0

        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            # Training loop
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

        # Compute the score for the current hyperparameters
        score = compute_score(opt)  # Replace with your evaluation metric function

        # Check if the current score is better than the previous best score
        if score > best_score:
            best_score = score
            best_hyperparameters = hyperparameters

    return best_hyperparameters, best_score


# Define the hyperparameters and their possible values for grid search
hyperparameters_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [16, 32],
    'niter': [50, 100, 200],
    # Add more hyperparameters and their values as needed
}

# Generate the grid of hyperparameter combinations
hyperparameter_combinations = list(itertools.product(*hyperparameters_grid.values()))

# Perform grid search
best_hyperparameters, best_score = perform_grid_search(hyperparameter_combinations)

# Print the best hyperparameters and score
print("Best hyperparameters:", best_hyperparameters)
print("Best score:", best_score)
