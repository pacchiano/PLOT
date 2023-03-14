from modsel_experiments import AlgorithmEnvironment

class MahalanobisEnvironment:

    def __init__(self, ):




def train_mahalanobis_modsel(dataset, baseline_model, num_batches, batch_size, 
    num_opt_steps, opt_batch_size,
    representation_layer_sizes = [10, 10], threshold = .5, 
    alphas = [1, .1, .01], lambda_reg = 1,
    verbose = False,
    restart_model_full_minimization = False, modselalgo = "Corral", 
    split = False, retraining_frequency = 1, 
    burn_in = -1):
    

    num_alphas = len(alphas)

    modsel_manager = get_modsel_manager(modselalgo, alphas, num_batches)

    alpha = alphas[0]
    ### THE above is going to fail for linear representations.


    (
        train_dataset,
        test_dataset,
    ) = get_dataset_simple(
        dataset=dataset,
        batch_size=batch_size,
        test_batch_size=10000000)


    dataset_dimension = train_dataset.dimension
    model_samples = [0 for _ in range(num_alphas)]

    if not split:


        model = TorchMultilayerRegressionMahalanobis(
            alpha=alpha,
            representation_layer_sizes=representation_layer_sizes,
            dim = train_dataset.dimension,
            output_filter = 'logistic'
        )

        growing_training_dataset = GrowingNumpyDataSet(num_batches, batch_size, train_dataset.dimension)

    if split:

        models = [ TorchMultilayerRegressionMahalanobis(
            alpha=alpha,
            representation_layer_sizes=representation_layer_sizes,
            dim = train_dataset.dimension,
            output_filter = 'logistic'
        ) for alpha in alphas]

        growing_training_datasets = [GrowingNumpyDataSet(num_batches, batch_size, train_dataset.dimension) for _ in range(num_alphas)]




    instantaneous_regrets = []
    instantaneous_accuracies = []
    modselect_info = []
    instantaneous_baseline_accuracies = []

    num_positives = []
    num_negatives = []
    false_neg_rates = []
    false_positive_rates = []

    optimistic_reward_predictions_list = []

    pessimistic_reward_predictions_list = []

    rewards = []



    if len(representation_layer_sizes) == 0:
        covariance  = lambda_reg*torch.eye(dataset_dimension)#.cuda()
    else:
        if torch.cuda.is_available():

            covariance = lambda_reg*torch.eye(representation_layer_sizes[-1])#.cuda()
        else:
            covariance = lambda_reg*torch.eye(representation_layer_sizes[-1])


    for i in range(num_batches):
        if verbose:
            print("Processing mahalanobis batch ", i)
        batch_X, batch_y = train_dataset.get_batch(batch_size)


            
        ### Sample epsilon using model selection
        sample_idx = modsel_manager.sample_base_index()
        model_samples[sample_idx] += 1
        alpha = alphas[sample_idx]
        print("Modselalgo {}".format(modselalgo))
        print(i, " batch number ", " sample alpha ", alpha)
        print("Alphas distribution ", modsel_manager.get_distribution())
        print("is split {}".format(split))
        if not split:
           model.alpha = alpha

        else:
            model = models[sample_idx]
            growing_training_dataset = growing_training_datasets[sample_idx]



        modselect_info.append(modsel_manager.get_distribution())

        with torch.no_grad():

            inverse_covariance = torch.linalg.inv(covariance)


            ##### Get thresholded predictions and uncertanties
            optimistic_thresholded_predictions, optimistic_prob_predictions, pessimistic_prob_predictions = model.get_all_predictions_info(batch_X, threshold, inverse_covariance)
         
            # IPython.embed()
            # raise ValueError("asdlfkm")

            if i <= burn_in:
                #IPython.embed()
                optimistic_thresholded_predictions = optimistic_thresholded_predictions >= -10


            baseline_predictions = baseline_model.get_thresholded_predictions(batch_X, threshold)


            boolean_labels_y = batch_y.bool().squeeze()
            accuracy = (torch.sum(optimistic_thresholded_predictions*boolean_labels_y) +torch.sum( ~optimistic_thresholded_predictions*~boolean_labels_y))*1.0/batch_size


            #### TOTAL NUM POSITIVES
            total_num_positives = torch.sum(optimistic_thresholded_predictions)

            #### TOTAL NUM NEGATIVES   
            total_num_negatives = torch.sum(~optimistic_thresholded_predictions)

            #### FALSE NEGATIVE RATE
            false_neg_rate = torch.sum(~optimistic_thresholded_predictions*boolean_labels_y)*1.0/(torch.sum(~optimistic_thresholded_predictions)+.00000000001)


            #### FALSE POSITIVE RATE            
            false_positive_rate = torch.sum(optimistic_thresholded_predictions*~boolean_labels_y)*1.0/(torch.sum(optimistic_thresholded_predictions)+.00000000001)
            



            num_positives.append(total_num_positives.item())
            num_negatives.append(total_num_negatives.item())
            false_neg_rates.append(false_neg_rate.item())
            false_positive_rates.append(false_positive_rate)            




            ### Update mod selection manager
            modsel_info = dict([])
            

            modesel_reward = (2*torch.sum(optimistic_thresholded_predictions*boolean_labels_y) - torch.sum(optimistic_thresholded_predictions))/batch_size
            rewards.append(modesel_reward)


            baseline_reward = (2*torch.sum(baseline_predictions*boolean_labels_y) - torch.sum(baseline_predictions))/batch_size



            optimistic_reward_predictions_list.append(((2*torch.sum(optimistic_thresholded_predictions*optimistic_prob_predictions) - torch.sum(optimistic_thresholded_predictions))/batch_size).item())
            modsel_info["optimistic_reward_predictions"] = optimistic_reward_predictions_list[-1]

            pessimistic_reward_predictions_list.append(((2*torch.sum(optimistic_thresholded_predictions*pessimistic_prob_predictions) - torch.sum(optimistic_thresholded_predictions))/batch_size).item())
            modsel_info["pessimistic_reward_predictions"] = pessimistic_reward_predictions_list[-1]

            modsel_manager.update_distribution(sample_idx, modesel_reward.item(), modsel_info )


            accuracy_baseline = (torch.sum(baseline_predictions*boolean_labels_y) +torch.sum( ~baseline_predictions*~boolean_labels_y))*1.0/batch_size
            #IPython.embed()

            instantaneous_baseline_accuracies.append(accuracy_baseline)

            instantaneous_regret = baseline_reward - modesel_reward
          
            print("Baseline Reward - {}".format(baseline_reward))
            print("Modsel Reward -   {}".format(modesel_reward))

            print("Baseline Accuracy - {}".format(accuracy_baseline))
            print("Accuracy alph reg - {}".format(accuracy))


            instantaneous_regrets.append(instantaneous_regret.item())
            instantaneous_accuracies.append(accuracy.item())


            filtered_batch_X = batch_X[optimistic_thresholded_predictions, :]
            
            ### Update the covariance
            filtered_representations_batch = model.get_representation(filtered_batch_X)
            covariance += torch.transpose(filtered_representations_batch, 0,1)@filtered_representations_batch

            filtered_batch_y = batch_y[optimistic_thresholded_predictions]



        growing_training_dataset.add_data(filtered_batch_X, filtered_batch_y)

        #### Filter the batch using the predictions
        #### Add the accepted points and their labels to the growing training dataset
        if i%retraining_frequency == 0 or (i== burn_in)*(split==False) or (model_samples[sample_idx] == burn_in)*split:
            #IPython.embed()


            print("Training Model --- iteration {}".format(i))
            model = train_model( model, num_opt_steps, 
                    growing_training_dataset, opt_batch_size, 
                    restart_model_full_minimization = restart_model_full_minimization)
     
                
    print("Finished training mahalanobis model alpha - {}".format(alpha))
    test_accuracy = evaluate_model(test_dataset, model, threshold).item()

    results = dict([])
    results["instantaneous_regrets"] = instantaneous_regrets
    results["test_accuracy"] = test_accuracy
    results["instantaneous_accuracies"] = instantaneous_accuracies

    results["instantaneous_baseline_accuracies"] = instantaneous_baseline_accuracies

    results["num_negatives"] = num_negatives
    results["num_positives"] = num_positives
    results["false_neg_rates"] = false_neg_rates
    results["false_positive_rates"] = false_positive_rates
    results["modselect_info"] = modselect_info

    results["rewards"] = rewards
    results["optimistic_reward_predictions"] = optimistic_reward_predictions_list
    results["pessimistic_reward_predictions"] = pessimistic_reward_predictions_list

    return results# instantaneous_regrets, instantaneous_accuracies, test_accuracy, modselect_info

