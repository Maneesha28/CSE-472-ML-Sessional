from train_1805076 import *

np.random.seed(42)

# independent_test_dataset = ds.EMNIST(root='./data', split='letters',
#                              train=False,
#                              transform=transforms.ToTensor())

with open('ids7.pickle', 'rb') as ids7:
  independent_test_dataset = pickle.load(ids7)

processed_test_data = data_processing(independent_test_dataset)
x_test, y_test = split_features_labels(processed_test_data)

if __name__ == "__main__":
    # load model
    with open('model_1805076.pickle', 'rb') as file:
        loaded_model = pickle.load(file)

    y_pred = loaded_model.predict(x_test)

    f1_score, accuracy, loss = loaded_model.performance_metrices(y_test, y_pred)

    print('F1 Score: {f1_score}, Accuracy: {accuracy}, Loss: {loss}'.format(f1_score=f1_score, accuracy=accuracy, loss=loss))