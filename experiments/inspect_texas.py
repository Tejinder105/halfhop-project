from halfhop.datasets import load_texas


dataset, data = load_texas()

print("Dataset:")
print(dataset)

print("\nGraph:")
print(data)

print("\nNumber of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Number of features:", dataset.num_features)
print("Number of classes:", dataset.num_classes)

print("\nNode features:")
print(data.x.shape)

print("\nLabels:")
print(data.y.shape)

print("\nTrain mask:")
print(data.train_mask.shape)

print("\nValidation mask:")
print(data.val_mask.shape)

print("\nTest mask:")
print(data.test_mask.shape)