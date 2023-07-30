from typing import List, Dict, TypeVar, Any, NamedTuple, Optional
from collections import defaultdict, Counter
import math
# Streamline all import
import csv


# Define Candidate
class Candidate:
  def __init__(self, no, ukuran, lantai, tarif_internet, tipe_bangunan, harga_sewa, kategori):
    self.no = int(no)

    # Call classification saat init value, namun hasil output berbeda.
    # namun saya kurang paham value mana yang benar
    # -------------------------------------------- Output --------------------------------------------
    # Without calling ukuran_classification(ukuran) : Entropy Split 'ukuran':  0.0
    #                                                 Information Gain 'ukuran':  3.321928094887362
    # When calling    ukuran_classification(ukuran) : Entropy Split 'ukuran':  1.4854752972273344
    #                                                 Information Gain 'ukuran':  1.8364527976600278
    # ------------------------------------------------------------------------------------------------
    # self.ukuran = int(ukuran)
    self.ukuran = self.ukuran_classification(ukuran)
    self.lantai = self.lantai_classification(lantai)
    self.tarif_internet = self.tarif_internet_classification(tarif_internet)
    self.tipe_bangunan = tipe_bangunan
    self.harga_sewa = self.harga_sewa_classification(harga_sewa)
    self.kategori = kategori

  def ukuran_classification(self, ukuran):
    # ukuran = self.ukuran
    if ukuran <= 200:
      return "kecil"
    elif 200 < ukuran <= 350:
      return "sedang"
    else:
      return "besar"

  def lantai_classification(self, lantai):
    # lantai = self.lantai
    if lantai <= 4:
      return "rendah"
    elif 4 < lantai <= 8:
      return "sedang"
    else:
      return "tinggi"

  def tarif_internet_classification(self, tarif_internet):
    # tarif_internet = self.tarif_internet
    if tarif_internet <= 8:
      return "rendah"
    elif 8 < tarif_internet <= 50:
      return "sedang"
    else:
      return "tinggi"

  def harga_sewa_classification(self, harga_sewa):
    # harga_sewa = self.harga_sewa
    if harga_sewa <= 400:
      return "rendah"
    elif 400 < harga_sewa <= 550:
      return "sedang"
    else:
      return "tinggi"


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



