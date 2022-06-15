import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from data.parameters import PATH, LOAD, RETRAIN, PRINT_INTERVAL, NUM_EPOCHS, BATCH_SIZE, NUM_WORKERS, SHUFFLE, NUM_CLASSES, CLASSES_MAPPING
from models.audio_model import audio_model
from data.build_dataset import build_datasets

"""
Validation Score: 78.67 %
Validation Score per Class:
	Target 0: 85.0 %
	Target 1: 78.9 %
	Target 2: 79.1 %
	Target 3: 81.7 %
	Target 4: 81.2 %
	Target 5: 71.1 %
	Target 6: 80.6 %
	Target 7: 73.6 %
	Target 8: 84.8 %
	Target 9: 61.0 %
	Target 10: 86.6 %
	Target 11: 51.3 %
	Target 12: 90.4 %
	Target 13: 89.7 %
	Target 14: 64.1 %
	Target 15: 81.0 %
	Target 16: 82.6 %
	Target 17: 84.8 %
	Target 18: 66.3 %
	Target 19: 78.0 %
	Target 20: 68.3 %
	Target 21: 86.3 %
	Target 22: 81.8 %
	Target 23: 86.6 %
	Target 24: 89.2 %
	Target 25: 79.6 %
	Target 26: 83.1 %
	Target 27: 70.8 %
	Target 28: 82.4 %
	Target 29: 88.1 %
	Target 30: 68.9 %
	Target 31: 87.8 %
	Target 32: 87.6 %
	Target 33: 84.4 %
	Target 34: 77.3 %
"""

class training_agent(
    ):
  
    def __init__(
        self,
        train_dataset,
        validation_dataset,
        save_path,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=SHUFFLE,
        num_classes=NUM_CLASSES,
        num_epochs=NUM_EPOCHS,
        print_interval=PRINT_INTERVAL
        ):
      
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dataset = train_dataset
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle
            )

        self.validation_dataset = validation_dataset

        self.num_classes = num_classes
        self.model = audio_model(num_classes=self.num_classes)
        self.model.to(self.device)

        loss_weight = torch.tensor([1.53, 1.31, 1.31, 1.31, 1.31, 0.66, 0.66,
                                    0.66, 1.75, 1.75, 0.66, 0.66, 1.31, 1.31,
                                    1.75, 0.66, 1.31, 0.66, 0.66, 0.66, 0.66,
                                    0.66, 0.66, 0.66, 1.31, 0.66, 0.66, 0.66,
                                    1.53, 0.66, 0.66, 1.75, 1.31, 0.66, 0.66])
        self.loss_function = torch.nn.NLLLoss(weight=loss_weight)
        self.loss_function.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=0.0001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.1)

        self.num_epochs = num_epochs
        self.print_interval = print_interval

        self.save_path = save_path
        self.logs = {
            "train_loss": [],
            "validation_score": []
        }

    def load(
        self
        ):
      
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, "model.pth"), map_location=self.device))
        with open(os.path.join(self.save_path, "logs.json"), "r") as f:
            self.logs = json.load(f)

    def train(
        self
        ):
      
        for epoch in range(1, self.num_epochs + 1):
            print(f"Epoch: {epoch}/{self.num_epochs}:")
            self.train_one_epoch()
            self.validate(save_flag=True)
            with open(os.path.join(self.save_path, "logs.json"), "w") as f:
                json.dump(self.logs, f)
        print("Done Training!")

    def train_one_epoch(
        self
        ):
      
        self.model.train()
        loss_log = []
        for train_step, batch in enumerate(self.train_dataloader):
            train_input, train_target = [t.to(self.device) for t in batch]
            output = self.model.forward(train_input.to(self.device))
            loss = self.loss_function(output.squeeze(), train_target.squeeze())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_log.append(loss.item())
            if train_step % self.print_interval == 0:
                print(f"\r\tTraining: {train_step}/{len(self.train_dataloader)} - Loss: {loss:3.5f}", end="")
        self.logs["train_loss"].append(sum(loss_log) / len(loss_log))
        self.scheduler.step()
  
    def validate(
        self,
        save_flag=False
        ):
      
        self.model.eval()
        with torch.no_grad():
          total = np.zeros(self.num_classes)
          correct = np.zeros(self.num_classes)
          for val_step in range(len(self.validation_dataset)):
              val_input, val_target = self.validation_dataset[val_step]
              val_input = val_input.view(1, val_input.shape[0], val_input.shape[1], val_input.shape[2])
              output = self.model.forward(val_input.to(self.device))
              target = val_target.item()
              prediction = torch.argmax(output.squeeze()).item()
              total[target] += 1
              if target == prediction:
                  correct[target] += 1
              if val_step % self.print_interval == 0:
                  print(f"\r\tValidation: {val_step}/{len(self.validation_dataset)}", end="")
          score = float(np.sum(correct) / np.sum(total) * 100)
          self.logs["validation_score"].append(score)
          if max(self.logs["validation_score"]) == score and save_flag:
              torch.save(self.model.state_dict(), os.path.join(self.save_path, "model.pth"))
          print(f"\r\tValidation Score: {score} %")
          print("\tValidation Score per Class:")
          score_per_class = []
          for i in range(len(total)):
              if total[i] != 0:
                  score_per_class.append(round(correct[i] / total[i] * 100, 1))
              else:
                  score_per_class.append("-")
              print(f"\t\tTarget {i}: {score_per_class[-1]} %")
          if max(self.logs["validation_score"]) == score:
              self.logs["validation_score_per_class"] = score_per_class

def plot_results(
    logs,
    classes_mapping=CLASSES_MAPPING
    ):
    
    epochs = list(range(1, len(logs["train_loss"]) + 1))
    
    plt.figure()
    plt.plot(epochs, logs["train_loss"], color=[0, 0.3, 1], linewidth=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train Loss per Epoch")
    
    plt.figure()
    plt.plot(epochs, logs["validation_score"], color=[0, 0.3, 1], linewidth=3)
    plt.xlabel("Epoch")
    plt.ylabel("Score (Correct Percentage)")
    plt.title("Validation Score per Epoch")
    
    labels = list(classes_mapping.keys())
    x = range(len(labels))
    
    plt.figure()
    plt.bar(x, logs["validation_score_per_class"], color=[0, 0.3, 1], width=0.5)
    plt.xticks(x, labels, rotation=90)
    plt.xlabel("Label")
    plt.ylabel("Score (Correct Percentage)")
    plt.title("Validation Score per Class")
    
def main(
    ):
    
    save_path = PATH
    train_dataset, validation_dataset = build_datasets()
    agent = training_agent(train_dataset=train_dataset, validation_dataset=validation_dataset, save_path=save_path)
    if os.path.isfile(os.path.join(save_path, "model.pth")) and LOAD:
        agent.load()
        if RETRAIN:
            agent.train()
        else:
            plot_results(agent.logs)
    else:
        agent.train()

if __name__ == "__main__":
    main()