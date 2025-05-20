# Generate 3-Layer Tree with N clients

```bash
./make_3layer_tree.py N P1 P2 P3
```

Paper Experiments
```bash
./make_3layer_tree.py 100 5 15 80
./make_3layer_tree.py 100 10 30 60
./make_3layer_tree.py 100 20 35 45
./make_3layer_tree.py 100 34 33 33 -o 3-layer-tree-33-33-33.json
./make_3layer_tree.py 100 45 35 20
./make_3layer_tree.py 100 60 30 10
./make_3layer_tree.py 100 80 0 20
./make_3layer_tree.py 100 80 15 5
```

# CIFAR-10 Non-IID Data Partition

Paper Experiments
```bash
./generate_data.py --n_clients 100 --by_labels_split --scenario equal --outfile configuration-cifar10-federated-3-layer-tree-eq.json
./generate_data.py --n_clients 100 --by_labels_split --scenario biased --outfile configuration-cifar10-federated-3-layer-tree-d_to_n-2x.json
./generate_data.py --n_clients 100 --by_labels_split --scenario highly_biased --outfile configuration-cifar10-federated-3-layer-tree-d_to_n-tiny_n1.json
```