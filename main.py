from typing import List, Dict, TypeVar, Any, NamedTuple, Optional
from collections import defaultdict, Counter
import math
# Streamline all import
import csv


# Define Candidate
class Candidate:
  def __init__(self, no, ukuran, lantai, tarif_internet, tipe_bangunan, harga_sewa, kategori):
    self.no = int(no)
    self.tipe_bangunan = tipe_bangunan
    self.kategori = kategori

    # Call classification saat init value, namun hasil output berbeda.
    # namun saya kurang paham value mana yang benar
    # -------------------------------------------- Output --------------------------------------------
    # Without calling ukuran_classification(ukuran) : Entropy Split 'ukuran':  0.0
    #                                                 Information Gain 'ukuran':  3.321928094887362
    # When calling    ukuran_classification(ukuran) : Entropy Split 'ukuran':  1.4854752972273344
    #                                                 Information Gain 'ukuran':  1.8364527976600278
    # ------------------------------------------------------------------------------------------------
    self.ukuran = int(ukuran)
    self.lantai = int(lantai)
    self.tarif_internet = int(tarif_internet)
    self.harga_sewa = int(harga_sewa)

    # self.ukuran = self.ukuran_classification()
    # self.lantai = self.lantai_classification()
    # self.tarif_internet = self.tarif_internet_classification()
    # self.harga_sewa = self.harga_sewa_classification()

  def ukuran_classification(self):
    # ukuran = self.ukuran
    if self.ukuran <= 200:
      return "kecil"
    elif 200 < self.ukuran <= 350:
      return "sedang"
    else:
      return "besar"

  def lantai_classification(self):
    # lantai = self.lantai
    if self.lantai <= 4:
      return "rendah"
    elif 4 < self.lantai <= 8:
      return "sedang"
    else:
      return "tinggi"

  def tarif_internet_classification(self):
    # tarif_internet = self.tarif_internet
    if self.tarif_internet <= 8:
      return "SohoA"
    elif 8 < self.tarif_internet <= 50:
      return "SohoB"
    else:
      return "SohoC"

  def harga_sewa_classification(self):
    # harga_sewa = self.harga_sewa
    if self.harga_sewa <= 400:
      return "murah"
    elif 400 < self.harga_sewa <= 550:
      return "menengah"
    else:
      return "mahal"


# Use Data.csv for Candidate Objects
def read_data_from_csv(file_path: str) -> List[Candidate]:
  candidates = []
  with open(file_path, 'r', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      candidate = Candidate(
        no=row['No'],
        ukuran=int(row['Ukuran']),
        lantai=int(row['Lantai']),
        tarif_internet=int(row['Tarif Internet']),
        tipe_bangunan=row['Tipe Bangunan'],
        harga_sewa=int(row['Harga Sewa']),
        kategori=row['Kategori']
      )
      candidates.append(candidate)
  return candidates


# Read Data.csv as input
data_file = "Data.csv"
inputs = read_data_from_csv(data_file)


# ---------------------------------------------------------
# Modified Code from Jupyter decision_tree.ipynb
# ---------------------------------------------------------
def entropy(class_probabilities: List[float]) -> float:
  """Given a list of class probabilities, compute the entropy"""
  return sum(-p * math.log(p, 2)
             for p in class_probabilities
             if p > 0)  # ignore zero probabilities


assert entropy([1.0]) == 0
assert entropy([0.5, 0.5]) == 1
assert 0.81 < entropy([0.25, 0.75]) < 0.82


def class_probabilities(labels: List[Any]) -> List[float]:
  total_count = len(labels)
  return [count / total_count
          for count in Counter(labels).values()]


def data_entropy(labels: List[Any]) -> float:
  return entropy(class_probabilities(labels))


# # Output Class Entropy
# print("Class Entropy: ", entropy(class_probabilities(inputs)))

# ---------------------------------------------------------
# Split Attribute
# ---------------------------------------------------------
T = TypeVar('T')  # generic type for inputs


def partition_entropy(subsets: List[List[Any]]) -> float:
  """Returns the entropy from this partition of data into subsets"""
  total_count = sum(len(subset) for subset in subsets)

  return sum(data_entropy(subset) * len(subset) / total_count
             for subset in subsets)


def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
  """Partition the inputs into lists based on the specified attribute."""
  partitions: Dict[Any, List[T]] = defaultdict(list)
  for input in inputs:
    key = getattr(input, attribute)  # value of the specified attribute
    partitions[key].append(input)    # add input to the correct partition
  return partitions


def partition_entropy_by(inputs: List[Any],
                         attribute: str,
                         label_attribute: str) -> float:
  """Compute the entropy corresponding to the given partition"""
  # partitions consist of our inputs
  partitions = partition_by(inputs, attribute)

  # but partition_entropy needs just the class labels
  labels = [[getattr(input, label_attribute) for input in partition]
            for partition in partitions.values()]

  return partition_entropy(labels)


# ---------------------------------------------------------
# Output
# ---------------------------------------------------------
# Output Class Entropy
print("Class Entropy: ", entropy(class_probabilities(inputs)))

# Output Entropy Split
attributes = ['ukuran', 'lantai', 'tarif_internet', 'tipe_bangunan', 'kategori']
for attribute in attributes:
  entropy_split = partition_entropy_by(inputs, attribute, 'harga_sewa')
  print(f"Entropy Split '{attribute}': ", entropy_split)

# Information Gain
for attribute in attributes:
  entropy_split = partition_entropy_by(inputs, attribute, 'harga_sewa')
  information_gain = entropy(class_probabilities(inputs)) - entropy_split
  print(f"Information Gain '{attribute}': ", information_gain)

# ---------------------------------------------------------
# Tree Output
# From test too heavy for replit, so not merged
# ---------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree, preprocessing


# Compute the classifications and store them as instance variables
# Alternative: use input.ukuran as int
for input in inputs:
  input.ukuran_class = input.ukuran_classification()
  input.lantai_class = input.lantai_classification()
  input.tarif_internet_class = input.tarif_internet_classification()
  input.harga_sewa_class = input.harga_sewa_classification()

# Create a DataFrame with the data and target labels
data = {
  'ukuran': [input.ukuran_class for input in inputs],
  'lantai': [input.lantai_class for input in inputs],
  'tarif_internet': [input.tarif_internet_class for input in inputs],
  'tipe_bangunan': [input.tipe_bangunan for input in inputs],
  'harga_sewa': [input.harga_sewa_class for input in inputs],
  # Cannot accept string, so use target labels
  # 'kategori': ['Biasa', 'Biasa', 'VIP', 'Biasa', 'Biasa', 'VIP', 'VIP', 'Eksklusif', 'Eksklusif', 'Eksklusif'],
  # ??? tidak tahu tiba tiba bisa menggunakan input ???
  'kategori': [input.kategori for input in inputs],
}

df = pd.DataFrame(data)

# Perform label encoding on the categorical features
label_encoders = {}
for feature in ['ukuran', 'lantai', 'tarif_internet', 'tipe_bangunan', 'harga_sewa', 'kategori']:
  label_encoders[feature] = preprocessing.LabelEncoder()
  df[feature] = label_encoders[feature].fit_transform(df[feature])


label_encoder = preprocessing.LabelEncoder()
df['kategori'] = label_encoder.fit_transform(df['kategori'])

# plot and Create tree model
X = df.drop('kategori', axis=1)
y = df['kategori']
model = tree.DecisionTreeClassifier()
model.fit(X, y)

# Output Decision Tree
plt.figure(figsize=(15, 20))
# tree.plot_tree(model, feature_names=list(X.columns), class_names=label_encoders['kategori'].classes_, filled=True)
# Fix use static value
# tree.plot_tree(model, feature_names=list(X.columns), class_names=['Biasa', 'Eksklusif', 'VIP'], filled=True)
# alt, use as list
tree.plot_tree(model, feature_names=list(X.columns), class_names=list(label_encoders['kategori'].classes_), filled=True)
# tree.plot_tree(model, feature_names=list(X.columns), class_names=list(label_encoder.classes_), filled=True)

plt.show()













