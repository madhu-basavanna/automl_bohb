import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

# plot training and validation accuracy and precision
# fig = plt.figure(figsize=(8, 4))
# acc, prec, loss, model_size = [], [], [], []
#
# with open('output_acc_prec.json', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         acc.append(data["top3_accuracy"])
#         prec.append(data["precision"])
#         loss.append(data['loss'])
#         model_size.append(data['model_size'])
#
# df = pd.DataFrame({'top2_accuracy': acc, 'precision': prec, 'loss': loss, 'model_size': model_size})
# df.sort_values(by='model_size')
#
# plt.plot(df.get('model_size'), df.get('top2_accuracy'), 'ro', label="Accuracy")
# plt.plot(df.get('model_size'), df.get('precision'), 'bo', label='Precision')
# plt.plot(df.get('model_size'), df.get('loss'), 'go', label='Loss')
# plt.legend(loc="upper right")
# plt.axis([20000, 100000, 0, 1])
# plt.title('Comparing model size with model performance')
# plt.xlabel('Network size')
# plt.ylabel('Performance of models')
#
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
# fig.savefig('performance_plot_0.png', bbox_inches='tight')
#
# # plot test accuracy and precision
# fig = plt.figure(figsize=(8, 4))
# acc, prec, loss, model_size = [], [], [], []
#
# with open('test_acc_prec.json', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         acc.append(data["top3_accuracy"])
#         prec.append(data["precision"])
#         model_size.append(data['model_size'])
#
# df = pd.DataFrame({'top2_accuracy': acc, 'precision': prec, 'model_size': model_size})
# df.sort_values(by='model_size')
#
# plt.plot(df.get('model_size'), df.get('top2_accuracy'), 'ro', label="Test Accuracy")
# plt.plot(df.get('model_size'), df.get('precision'), 'bo', label='Test Precision')
# plt.legend(loc="upper right")
# plt.axis([20000, 100000, 0, 1])
# plt.title('Comparing model size with model performance')
# plt.xlabel('Network size')
# plt.ylabel('Performance of models')
#
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
# fig.savefig('test_performance_plot_0.png', bbox_inches='tight')


# # plot test accuracy and precision without NAS and HPO
# fig = plt.figure(figsize=(8, 4))
# acc, prec, data_subset, model_size = [], [], [], []
#
# with open('main_output_all.json', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         acc.append(data["test_accuracy"])
#         prec.append(data["test_precision"])
#         model_size.append(data['model_size'])
#         data_subset.append(data['data_subset'])
#
# df = pd.DataFrame({'top2_accuracy': acc, 'precision': prec, 'model_size': model_size, 'data_subset': data_subset})
# df.sort_values(by='data_subset')
#
# plt.plot(df.get('data_subset'), df.get('top2_accuracy'), 'ro', label="Test Accuracy")
# plt.plot(df.get('data_subset'), df.get('precision'), 'bo', label='Test Precision')
# plt.legend(loc="upper right")
# plt.title('Comparing model performance with different subset of the data')
# plt.xlabel('Index of data subset')
# plt.ylabel('Performance of models')
#
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
# fig.savefig('main_test_performance_plot_0.png', bbox_inches='tight')



# plot learning curve
fig = plt.figure(figsize=(8, 4))
acc, prec, time_sec = [], [], []

with open('learning_curve.json', 'r') as f:
    for line in f:
        data = json.loads(line)
        acc.append(data["accuracy"])
        prec.append(data["precision"])
        time_sec.append(data['time_sec'])

df = pd.DataFrame({'top2_accuracy': acc, 'precision': prec, 'time_sec': time_sec})
#df.sort_values(by='data_subset')

plt.plot(df.get('time_sec'), df.get('top2_accuracy'), 'r', label="Test Accuracy")
plt.plot(df.get('time_sec'), df.get('precision'), 'b', label='Test Precision')
plt.legend(loc="upper right")
plt.title('Learning curve')
plt.xlabel('Time in seconds')
plt.ylabel('Performance of models')

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('learning curve.png', bbox_inches='tight')