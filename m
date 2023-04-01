def siamese_training(training_dataloader, validation_dataloader, output_folder_name, learning_rate = 0.000005):
    
    net = SiameseNetwork101().cuda()
    criterion = ContrastiveLoss()
    criterion.margin = 2.0
    optimizer = optim.Adam(net.parameters(),lr = learning_rate)

    num_epochs = 1000
    training_losses, validation_losses = [], []
    training_accuracies, validation_accuracies = [], []
    euclidean_distance_threshold = 1

    epochs_no_improve = 0
    max_epochs_stop = 3 # "patience" - number of epochs with no improvement in validation loss after which training stops
    validation_loss_min = np.Inf
    validation_max_accuracy = 0
    history = []
'''
    output_dir = working_path + output_folder_name
    os.mkdir(output_dir)
    f = open(output_dir + "/history_training.txt", 'a+')
    f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" + "Training starting now...\n")
    f.close()
'''
    for epoch in range(0, num_epochs):
        print("epoch training started...")

        training_loss = 0
        validation_loss = 0

        training_accuracy_history = []
        validation_accuracy_history = []

        training_euclidean_distance_history = []
        training_label_history = []
        validation_euclidean_distance_history = []
        validation_label_history = []
        feature_map = []

        net.train()
          
        for i, data in enumerate(training_dataloader, 0):
            
            img0, img1, label, meta = data
            img0 = np.repeat(img0, 3, 1) # repeat grayscale image in 3 channels (ResNet requires 3-channel input) 
            img1 = np.repeat(img1, 3, 1)
            img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()  # send tensors to the GPU
            optimizer.zero_grad()
            output0, output1 = net.forward(img0, img1)
            loss_contrastive = criterion(output0, output1, label.float())
            loss_contrastive.backward()
            optimizer.step()

            
            training_loss += loss_contrastive.item()

            net.eval()
            output0, output1 = net.forward(img0, img1)
            net.train()
            euclidean_distance = F.pairwise_distance(output0, output1)
            training_label = euclidean_distance > euclidean_distance_threshold # 0 if same, 1 if not same (progression) 
            equals = training_label.int() == label.int() # 1 if true
            acc_tmp = torch.Tensor.numpy(equals.cpu())
            training_accuracy_history.extend(acc_tmp)

            euclid_tmp = torch.Tensor.numpy(euclidean_distance.detach().cpu()) # detach gradient, move to CPU
            training_euclidean_distance_history.extend(euclid_tmp)
            label_tmp = torch.Tensor.numpy(label.cpu())
            training_label_history.extend(label_tmp)
            print("training loop " + str(i) + " completed")

        else:
            print("validation started...")

            with torch.no_grad(): 
                net.eval()

                for j, data2 in enumerate(validation_dataloader, 0):
                    img0, img1, label, meta = data2
                    img0 = np.repeat(img0, 3, 1) # repeat grayscale image in 3 channels (ResNet requires 3-channel input) 
                    img1 = np.repeat(img1, 3, 1)
                    img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
                    output1, output2 = net.forward(img0, img1)
                    loss_contrastive = criterion(output1, output2, label.float())
                    validation_loss += loss_contrastive.item()
                     
                    euclidean_distance = F.pairwise_distance(output1, output2)
                    validation_label = euclidean_distance > euclidean_distance_threshold # 0 if same, 1 if not same
                    equals = validation_label.int() == label.int() # 1 if true
                    acc_tmp = torch.Tensor.numpy(equals.cpu())
                    validation_accuracy_history.extend(acc_tmp)
                    
                    euclid_tmp = torch.Tensor.numpy(euclidean_distance.detach().cpu()) # detach gradient, move to CPU
                    validation_euclidean_distance_history.extend(euclid_tmp)
                    label_tmp = torch.Tensor.numpy(label.cpu())
                    validation_label_history.extend(label_tmp)

                    print("validation loop " + str(j) + " completed")

            training_loss_avg = training_loss/len(training_dataloader)
            validation_loss_avg = validation_loss/len(validation_dataloader)

            training_accuracy = statistics.mean(np.array(training_accuracy_history).tolist())
            validation_accuracy = statistics.mean(np.array(validation_accuracy_history).tolist())


            if validation_loss_avg < validation_loss_min:
                # save model
                torch.save(net.state_dict(), output_dir + "/siamese_EAR_model.pth")
                # track improvement
                epochs_no_improve = 0
                validation_loss_min = validation_loss_avg
                validation_max_accuracy = validation_accuracy
                best_epoch = epoch
                

 
            # Otherwise increment count of epochs with no improvement
            else: 
                epochs_no_improve += 1 
                # Trigger EARLY STOPPING
                if epochs_no_improve >= max_epochs_stop:
                    print(f'\nEarly Stopping! Total epochs (starting from 0): {epoch}. Best epoch: {best_epoch} with loss: {validation_loss_min:.2f} and acc: {100 * validation_max_accuracy:.2f}%')
                    # Load the best state dict (at the early stopping point)
                    net.load_state_dict(torch.load(output_dir + "/siamese_EAR_model.pth"))
                    # attach the optimizer
                    net.optimizer = optimizer

                    # save history with pickle
                    with open(output_dir + "/history_training.pckl", "wb") as f:
                        pickle.dump(history, f)

                    f = open(output_dir + "/history_training.txt", 'a+')
                    f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" +
                            "Early stopping! Total epochs (starting from 0): {:.0f}\n".format(epoch) +
                            "Best epoch: {:.0f}\n".format(best_epoch) +
                            "Validation loss at best epoch: {:.3f}\n".format(validation_loss_min) +
                            "Validation accuracy at best epoch: {:3f}\n".format(validation_max_accuracy)
                            )
                    f.close()

                    return net, history # break the function

        # append to lists for graphing
        training_losses.append(training_loss_avg)
        validation_losses.append(validation_loss_avg)
        training_accuracies.append(training_accuracy)
        validation_accuracies.append(validation_accuracy)

        # training euclidean distance stats
        # extract euclidean distances if label is 0 or 1
        euclid_if_0 = [b for a, b in zip(training_label_history, training_euclidean_distance_history) if a == 0]
        euclid_if_1 = [b for a, b in zip(training_label_history, training_euclidean_distance_history) if a == 1]
        euclid_if_0 = np.array(euclid_if_0).tolist()
        euclid_if_1 = np.array(euclid_if_1).tolist()
        
        # summary statistics for euclidean distances
        mean_euclid_0t = statistics.mean(euclid_if_0) 
        std_euclid_0t = statistics.pstdev(euclid_if_0) # population stdev
        mean_euclid_1t = statistics.mean(euclid_if_1)
        std_euclid_1t = statistics.pstdev(euclid_if_1) # population stdev
        euclid_diff_t = mean_euclid_1t - mean_euclid_0t

        # validation euclidean distance stats
        # extract euclidean distances if label is 0 or 1
        euclid_if_0 = [b for a, b in zip(validation_label_history, validation_euclidean_distance_history) if a == 0]
        euclid_if_1 = [b for a, b in zip(validation_label_history, validation_euclidean_distance_history) if a == 1]
        euclid_if_0 = np.array(euclid_if_0).tolist()
        euclid_if_1 = np.array(euclid_if_1).tolist()
        
        # summary statistics for euclidean distances
        mean_euclid_0v = statistics.mean(euclid_if_0) 
        std_euclid_0v = statistics.pstdev(euclid_if_0) # population stdev
        mean_euclid_1v = statistics.mean(euclid_if_1)
        std_euclid_1v = statistics.pstdev(euclid_if_1) # population stdev
        euclid_diff_v = mean_euclid_1v - mean_euclid_0v

        # after the epoch is completed, adjust the euclidean_distance_threshold based on the validation mean euclidean distances
        euclidean_distance_threshold = (mean_euclid_0v + mean_euclid_1v) / 2

        # store in history list --> add the other euclidean stats here for graphs?
        history = [training_losses, validation_losses, training_accuracies, validation_accuracies,
                   euclid_diff_t, euclid_diff_v]
 
        # save history with pickle
        with open(output_dir + "/history_training.pckl", "wb") as f:
            pickle.dump(history, f)
 
        print("Epoch number: {:.0f}\n".format(epoch),
            "Training loss: {:.3f}\n".format(training_loss_avg),
            "Training accuracy: {:.3f}\n".format(training_accuracy),
            "Validation loss: {:.3f}\n".format(validation_loss_avg),
            "Validation accuracy: {:.3f}\n".format(validation_accuracy),
            "\nTraining \nLabel0 euclidean distance mean: {:.3f}\n".format(mean_euclid_0t),
            "Label0 euclidean distance std: {:.3f}\n".format(std_euclid_0t),
            "Label1 euclidean distance mean: {:.3f}\n".format(mean_euclid_1t),
            "Label1 euclidean distance std: {:.3f}\n".format(std_euclid_1t),
            "Euclidean distance mean diff: {:.3f}\n".format(euclid_diff_t),
            "\nValidation \nLabel0 euclidean distance mean: {:.3f}\n".format(mean_euclid_0v),
            "Label0 euclidean distance std: {:.3f}\n".format(std_euclid_0v),
            "Label1 euclidean distance mean: {:.3f}\n".format(mean_euclid_1v),
            "Label1 euclidean distance std: {:.3f}\n".format(std_euclid_1v),
            "Euclidean distance mean diff: {:.3f}\n".format(euclid_diff_v),
            "Euclidean distance threshold update: {:.3f}\n".format(euclidean_distance_threshold)
            )

        # write history to file
        f = open(output_dir + "/history_training.txt", 'a+')
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" +
            "Epoch number: {:.0f}\n".format(epoch) +
            "Training loss: {:.3f}\n".format(training_loss_avg) +
            "Training accuracy: {:.3f}\n".format(training_accuracy) +
            "Validation loss: {:.3f}\n".format(validation_loss_avg) +
            "Validation accuracy: {:.3f}\n".format(validation_accuracy) +
            "\nTraining \nLabel0 euclidean distance mean: {:.3f}\n".format(mean_euclid_0t) +
            "Label0 euclidean distance std: {:.3f}\n".format(std_euclid_0t) +
            "Label1 euclidean distance mean: {:.3f}\n".format(mean_euclid_1t) +
            "Label1 euclidean distance std: {:.3f}\n".format(std_euclid_1t) +
            "Euclidean distance mean diff: {:.3f}\n".format(euclid_diff_t) +
            "\nValidation \nLabel0 euclidean distance mean: {:.3f}\n".format(mean_euclid_0v) + 
            "Label0 euclidean distance std: {:.3f}\n".format(std_euclid_0v) + 
            "Label1 euclidean distance mean: {:.3f}\n".format(mean_euclid_1v) + 
            "Label1 euclidean distance std: {:.3f}\n".format(std_euclid_1v) +
            "Euclidean distance mean diff: {:.3f}\n".format(euclid_diff_v) +
            "Euclidean distance threshold update: {:.3f}\n".format(euclidean_distance_threshold) + "\n"
            )
        f.close()
 
    # Load the best state dict (at the early stopping point)
    net.load_state_dict(torch.load(output_dir + "/siamese_EAR_model.pth"))
    # After training through all epochs attach the optimizer
    net.optimizer = optimizer

    # Return the best model and history
    print(f'\nAll Epochs completed! Total epochs (starting from 0): {epoch}. Best epoch: {best_epoch} with validation loss: {validation_loss_min:.2f} and acc: {100 * validation_max_accuracy:.2f}%')
    return net, history, feature_map
 
# siamese training 
net, history,f_map = siamese_training(training_dataloader = training_dataloader, 
                               validation_dataloader = validation_dataloader, 
                               output_folder_name = 'Res101_inter')

# Training/validation learning curves
plt.title("Number of Training Epochs vs. Contrastive Loss")
plt.xlabel("Training Epochs")
plt.ylabel("Contrastive Loss")
plt.plot(range(0, len(history[0])), history[0], label = "Training loss")
plt.plot(range(0, len(history[1])), history[1], label = "Validation loss")
plt.legend(frameon=False)
plt.savefig(output_dir + "/Learning_curve.png")
plt.close()

