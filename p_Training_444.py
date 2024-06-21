import torch
import time
import Subfunction
import torch.utils.data as Data

model_name = '0825_2'
note = 'ID24 10 to 90, test, 1400K patch 444, change normalization, learning rate 0.0005, separate_input 50'
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
print('model name')
print(model_name)
print(note)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Training process
learning_rate = 0.0005
Iter_num = 1000
normalization_factor = 100000
testing_single_length = 200000
separate_input = 50
data_type = '10_to_90_444_resize'
training_patient = 'ID24'
testing_patient = ['2441', '2443']
testing_length = len(testing_patient) * testing_single_length
data_path = '/N/lustre/scratch/hancg/c_Under12/b_Training_testing_data/training_data/'
testing_input = torch.zeros([testing_length, 128])
testing_label = torch.zeros([testing_length, 64])
for i in range(len(testing_patient)):
    testing_input[i*testing_single_length:(i+1)*testing_single_length, :] = torch.load(data_path + testing_patient[i] + '/input_20_to_90_444_resize.pt')
    testing_label[i * testing_single_length:(i + 1) * testing_single_length, :] = torch.load(data_path + testing_patient[i] + '/target_20_to_90_444_resize.pt')


training_input = torch.load(data_path + training_patient + '/input_' + data_type + '.pt')
training_label = torch.load(data_path + training_patient + '/target_' + data_type + '.pt')

training_input, training_label = Subfunction.training_nor_2(training_input, training_label, normalization_factor)
testing_input, testing_label = Subfunction.training_nor_2(testing_input, testing_label, normalization_factor)

train_dataset = Data.TensorDataset(training_input, training_label)
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=int(testing_length/separate_input), shuffle=True,
                               num_workers=4)

network = Subfunction.network_128to64().to(device)

starttime = time.time()
print('Model structure')
print(network)

loss_fn = torch.nn.MSELoss(reduction='mean')
print('Loss function', loss_fn)
loss_fn_test = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, eps=3e-08)


for t in range(Iter_num):
    loss_print = 0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        training_pred = network(inputs)
        loss = loss_fn(training_pred, labels)
        loss_print += loss.item()
        loss.backward()
        optimizer.step()
    loss_print = loss_print / separate_input

    if True:
        with torch.no_grad():
            testing_input = testing_input.to(device)
            testing_label = testing_label.to(device)
            testing_pred = network(testing_input)
            loss_test2 = loss_fn_test(testing_pred, testing_label)
            print(t, loss_print, loss_test2.item())
    if t % 20 == 0:
        torch.save(network.state_dict(), 'ANN_model/' + model_name + '_' + str(t) + 'th_iter.pth')

print('Training success')
middletime = time.time()
print('Time', (middletime - starttime) / 60)




