import csv
import time

class Record:
    def __init__(self):
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []
        self.comm_costs = []
        self.timestamps = []
        self.cnt = 0

    def add(self, train_loss, train_acc, test_loss, test_acc, comm_cost=0.0):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.test_losses.append(test_loss)
        self.test_accs.append(test_acc)
        self.comm_costs.append(comm_cost)
        self.timestamps.append(time.time())
        self.cnt += 1

    def saveto(self, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_acc", "test_loss", "test_acc", "comm_cost", "timestamp"
            ])

            for i in range(self.cnt):
                writer.writerow([
                    i+1,
                    self.train_losses[i],self.train_accs[i],
                    self.test_losses[i],self.test_accs[i],
                    self.comm_costs[i],
                    self.timestamps[i]
                ])