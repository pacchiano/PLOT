import torch
import IPython

#@title Useful model training utilities
#Useful utilities. The following function allows to train a model 
### for a number of steps.
def gradient_step(model, optimizer, batch_X, batch_y):

    optimizer.zero_grad()
    loss = model.get_loss(batch_X, batch_y)
    loss.backward()
    optimizer.step()

    return model, optimizer

def train_model(
    model,
    num_steps,
    train_dataset,
    batch_size,
    verbose=False,
    restart_model_full_minimization=False,
    weight_decay=0.0
):
    if restart_model_full_minimization: 
        model.reset_weights()

    optimizer = torch.optim.Adam(model.network.parameters(), lr=0.01, weight_decay=weight_decay )

    for i in range(num_steps):
        if verbose:
            print("train model iteration ", i)
        batch_X, batch_y = train_dataset.get_batch(batch_size)

        model, optimizer = gradient_step(model, optimizer, batch_X, batch_y)

    return model


def train_model_opt_reg(
    model,
    num_steps,
    train_dataset,
    query_batch,
    batch_size,
    opt_reg = 1.0,
    verbose = False,    
    restart_model_full_minimization=False,
    weight_decay=0.0,
    log_optimism_loss = False
    ):
    

    if restart_model_full_minimization: 
        model.reset_weights()

    optimizer = torch.optim.Adam(model.network.parameters(), lr=0.01, weight_decay=weight_decay )

    for i in range(num_steps):
        if verbose:
            print("train model iteration ", i)
        batch_X, batch_y = train_dataset.get_batch(batch_size)


        

        #IPython.embed()
        ### We only evaluate the loss when we have collected some data
        if len(batch_X) > 0:
            # IPython.embed()
            # print("asdlfkmasdlfkmasdlfkm loss ")

            optimizer.zero_grad()

            loss = model.get_loss(batch_X, batch_y)
        


            predictions = model.predict(query_batch)


            ### Add the optimism regularizer
            #print("predictions   .... ", predictions)
            if log_optimism_loss:
                loss -= opt_reg*torch.mean(torch.log(predictions + .00000000001))

            else:
                loss -= opt_reg*torch.mean(predictions)

            #IPython.embed()

            loss.backward()
            optimizer.step()


    return model








def evaluate_model(test_dataset, model, threshold):
    with torch.no_grad():
        batch_test = test_dataset.get_batch(10000000000) 
        batch_X, batch_y = batch_test
        test_accuracy = model.get_accuracy(batch_X, batch_y, threshold)

    print("Final test model accuracy {}".format(test_accuracy))
    return test_accuracy

