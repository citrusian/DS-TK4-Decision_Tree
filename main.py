from typing import List, Dict, TypeVar, Any, NamedTuple, Optional
from collections import defaultdict, Counter
import math
# Streamline all import
import csv

# Define Candidate
class Candidate:
  def __init__(self, no, ukuran, lantai, tarif_internet, tipe_bangunan, harga_sewa, kategori):
    self.no = int(no)
    self.ukuran = int(ukuran)
    self.lantai = int(lantai)
    self.tarif_internet = int(tarif_internet)
    self.tipe_bangunan = tipe_bangunan
    self.harga_sewa = int(harga_sewa)
    self.kategori = kategori


  def ukuran_classification(self):
    ukuran = self.ukuran
    if ukuran <= 200:
      return "kecil"
    elif 200 < ukuran <= 350:
      return "sedang"
    else:
      return "besar"

  def lantai_classification(self):
    lantai = self.lantai
    if lantai <= 4:
      return "rendah"
    elif 4 < lantai <= 8:
      return "sedang"
    else:
      return "tinggi"

  def tarif_internet_classification(self):
    tarif_internet = self.tarif_internet
    if tarif_internet <= 8:
      return "rendah"
    elif 8 < tarif_internet <= 50:
      return "sedang"
    else:
      return "tinggi"

  def harga_sewa_classification(self):
    harga_sewa = self.harga_sewa
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
        ukuran=row['Ukuran'],
        lantai=row['Lantai'],
        tarif_internet=row['Tarif Internet'],
        tipe_bangunan=row['Tipe Bangunan'],
        harga_sewa=row['Harga Sewa'],
        kategori=row['Kategori']  # Fix, katagori terlewat masuk csv
      )
      candidates.append(candidate)
  return candidates

# Read Data.csv as input
data_file = "Data.csv"
inputs = read_data_from_csv(data_file)










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


assert data_entropy(['a']) == 0
assert data_entropy([True, False]) == 1
assert data_entropy([3, 4, 4, 4]) == entropy([0.25, 0.75])

entropy(class_probabilities([4, 4, 4, 4]))


def partition_entropy(subsets: List[List[Any]]) -> float:
  """Returns the entropy from this partition of data into subsets"""
  total_count = sum(len(subset) for subset in subsets)

  return sum(data_entropy(subset) * len(subset) / total_count
             for subset in subsets)


partition_entropy([[4, 4, 4, 4], [3, 2, 1]])








class Candidate(NamedTuple):
  level: str
  lang: str
  tweets: bool
  phd: bool
  did_well: Optional[bool] = None  # allow unlabeled data

  #  level     lang     tweets  phd  did_well
inputs = [Candidate('Senior', 'Java',   False, False, False),
          Candidate('Senior', 'Java',   False, True,  False),
          Candidate('Mid',    'Python', False, False, True),
          Candidate('Junior', 'Python', False, False, True),
          Candidate('Junior', 'R',      True,  False, True),
          Candidate('Junior', 'R',      True,  True,  False),
          Candidate('Mid',    'R',      True,  True,  True),
          Candidate('Senior', 'Python', False, False, False),
          Candidate('Senior', 'R',      True,  False, True),
          Candidate('Junior', 'Python', True,  False, True),
          Candidate('Senior', 'Python', True,  True,  True),
          Candidate('Mid',    'Python', False, True,  True),
          Candidate('Mid',    'Java',   True,  False, True),
          Candidate('Junior', 'Python', False, True,  False)
          ]




T = TypeVar('T')  # generic type for inputs

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

for key in ['level','lang','tweets','phd']:
  print(key, partition_entropy_by(inputs, key, 'did_well'))



senior_inputs = [input for input in inputs if input.level == 'Senior']

senior_inputs

assert 0.4 == partition_entropy_by(senior_inputs, 'lang', 'did_well')
assert 0.0 == partition_entropy_by(senior_inputs, 'tweets', 'did_well')
assert 0.95 < partition_entropy_by(senior_inputs, 'phd', 'did_well') < 0.96

junior_inputs = [input for input in inputs if input.level == 'Junior']

for key in ['lang', 'tweets', 'phd']:
  print(key, partition_entropy_by(junior_inputs, key, 'did_well'))












