#!/usr/bin/env python

import torch
import time

print("CUDA Available: ", torch.cuda.is_available())
print("Num CUDA Devices Available: ", torch.cuda.device_count())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

matrix1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device=device)
matrix2_inv = torch.inverse(matrix1)
product = torch.matmul(matrix1, matrix2_inv)

t0 = time.time()
result = product.cpu().numpy() #Move result to cpu before printing
t1 = time.time()
print("result of matrix multiplication")
print("===============================")
print(result)
print("Time: ", t1-t0)
print("===============================")
