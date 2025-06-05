# tsetlin_machine_binary.py
#
# binary classification
# based on TsetlinMachine.pyx (Pyrex)
# from github.com/cair/TsetlinMachine/tree/master

import numpy as np

class TsetlinMachine:
  # binary classifier
  def __init__(self, n_clauses, n_features,
    n_states, s, threshold, seed=0):

    self.n_clauses = n_clauses
    self.n_features = n_features
    self.n_states = n_states
    self.s = s                  # inverse update freq
    self.threshold = threshold  # voting min/max

    self.rnd = np.random.RandomState(seed)

    self.ta_state = self.rnd.choice([self.n_states,
      self.n_states+1], size=(self.n_clauses,
      self.n_features, 2)).astype(dtype=np.int32)

    # print(self.ta_state); input()

    self.clause_output = \
      np.zeros(shape=self.n_clauses, dtype=np.int32)
    self.feedback_to_clauses = \
      np.zeros(shape=self.n_clauses, dtype=np.int32)
    self.clause_sign = np.zeros(self.n_clauses,
      dtype=np.int32)
    for j in range(self.n_clauses):
      if j % 2 == 0:
        self.clause_sign[j] = 1
      else:
        self.clause_sign[j] = -1

# -----------------------------------------------------------

  def calculate_clause_output(self, x):
    # each cell is 0 or 1
    for j in range(self.n_clauses):
      self.clause_output[j] = 1
      for k in range(self.n_features):
        # action_include = self.action(self.ta_state[j,k,0])
        # action_include_negated = \
        #   self.action(self.ta_state[j,k,1])
        if self.ta_state[j,k,0] <= self.n_states:
          action_include = 0
        else:
          action_include = 1
        if self.ta_state[j,k,1] <= self.n_states:
          action_exclude = 0
        else:
          action_exclude = 1

        if (action_include == 1 and x[k] == 0) or \
           (action_exclude == 1 and x[k] == 1):
          self.clause_output[j] = 0
          break

# -----------------------------------------------------------

  def predict(self, x):
    self.calculate_clause_output(x)
    output_sum = self.sum_clause_votes() # -thresh to +thresh
    if output_sum >= 0:
      return 1
    else:
      return 0

# -----------------------------------------------------------

  # def action(self, state):
  #   if state <= self.n_states:
  #     return 0
  #   else:
  #     return 1

# -----------------------------------------------------------

  # def get_state(self, clause, feature, automaton_type):
  #   return self.ta_state[clause,feature,automaton_type]

# -----------------------------------------------------------

  def sum_clause_votes(self):
    # value between -thresh and +thresh
    output_sum = 0
    for j in range(self.n_clauses):
      output_sum += self.clause_output[j] * \
      self.clause_sign[j]

    if output_sum > self.threshold:
      output_sum = self.threshold
    elif output_sum < -self.threshold:
      output_sum = -self.threshold

    return output_sum

# -----------------------------------------------------------
# evaluate() replaced by accuracy()
# -----------------------------------------------------------
#   def evaluate(self, X, y):
#     n_examples = len(X)
#     xi = np.zeros((self.n_features,), dtype=np.int32)
#     errors = 0
#     for l in range(n_examples):
#       for j in range(self.n_features):
#         xi[j] = X[l,j]
#       self.calculate_clause_output(xi)
#       output_sum = self.sum_clause_votes()
#       if output_sum >= 0 and y[l] == 0:
#         errors += 1
#       elif output_sum < 0 and y[l] == 1:
#         errors += 1
# 
#     return 1.0 - ((1.0 * errors) / n_examples)

# -----------------------------------------------------------

  def accuracy(self, X, y):
    n_correct = 0; n_wrong = 0
    for i in range(len(X)):
      xi = X[i]
      actual_y = y[i]            # 0 or 1
      pred_y = self.predict(xi)  # 0 or 1
      if actual_y == pred_y:
        n_correct += 1
      else:
        n_wrong += 1
    return (n_correct * 1.0) / (n_correct + n_wrong)

# -----------------------------------------------------------

  def update(self, x, y):
    # worker for fit()
    self.calculate_clause_output(x)  # each cell 0 or 1
    output_sum = self.sum_clause_votes()  # -thresh, +thresh
    for j in range(self.n_clauses):
      self.feedback_to_clauses[j] = 0  # -1 or +1

    if y == 1:
      for j in range(self.n_clauses):
        if self.rnd.rand() > 1.0 * (self.threshold - \
          output_sum) / (2 * self.threshold):
          continue
        if self.clause_sign[j] == 1:
          self.feedback_to_clauses[j] = 1 # Type I Feedback
        else:
          self.feedback_to_clauses[j] = -1  # Type II
    elif y == 0:
      for j in range(self.n_clauses):
        if self.rnd.rand() > 1.0 * (self.threshold + \
          output_sum) / (2 * self.threshold):
          continue
        if self.clause_sign[j] == 1:
          self.feedback_to_clauses[j] = -1  # Type II
        else:
          self.feedback_to_clauses[j] = 1 # Type I

    for j in range(self.n_clauses):

      if self.feedback_to_clauses[j] == 1:
        if self.clause_output[j] == 0:
          for k in range(self.n_features):
            if self.rnd.rand() <= 1.0 / self.s:
              if self.ta_state[j,k,0] > 1:
                self.ta_state[j,k,0] -= 1
            if self.rnd.rand() <= 1.0 / self.s:
              if self.ta_state[j,k,1] > 1:
                self.ta_state[j,k,1] -= 1

        elif self.clause_output[j] == 1:
          for k in range(self.n_features):
            if x[k] == 1:
              if self.rnd.rand() <= \
              1.0 * (self.s-1) / self.s:
                if self.ta_state[j,k,0] < \
                self.n_states * 2:
                  self.ta_state[j,k,0] += 1
              if self.rnd.rand() <= 1.0 / self.s:
                if self.ta_state[j,k,1] > 1:
                  self.ta_state[j,k,1] -= 1
            elif x[k] == 0:
              if self.rnd.rand() <= \
              1.0 * (self.s-1) / self.s:
                if self.ta_state[j,k,1] < \
                self.n_states*2:
                  self.ta_state[j,k,1] += 1
              if self.rnd.rand() <= 1.0 / self.s:
                if self.ta_state[j,k,0] > 1:
                  self.ta_state[j,k,0] -= 1

      elif self.feedback_to_clauses[j] == -1:
        if self.clause_output[j] == 1:
          for k in range(self.n_features):

            # action_include = \
            #   self.action(self.ta_state[j,k,0])
            # action_include_negated = \
            #   self.action(self.ta_state[j,k,1])

            if self.ta_state[j,k,0] <= self.n_states:
              action_include = 0
            else:
              action_include = 1
            if self.ta_state[j,k,1] <= self.n_states:
              action_exclude = 0
            else:
              action_exclude = 1

            if x[k] == 0:
              if action_include == 0 and \
              self.ta_state[j,k,0] < self.n_states * 2:
                self.ta_state[j,k,0] += 1
            elif x[k] == 1:
              if action_exclude== 0 and \
              self.ta_state[j,k,1] < self.n_states * 2:
                self.ta_state[j,k,1] += 1

# -----------------------------------------------------------

  def fit(self, X, y, max_epochs):
    n_examples = len(X)
    xi = np.zeros(self.n_features, dtype=np.int32)
    # xi = np.zeros((self.n_features,), dtype=np.int32)
    indices = np.arange(n_examples)
    for epoch in range(max_epochs):
      if (epoch % 10 == 0): print(". ", flush=True, end="")
      self.rnd.shuffle(indices)
      for i in range(n_examples):
        example_id = indices[i]
        target_y = y[example_id]
        for j in range(self.n_features):
          xi[j] = X[example_id,j]
          self.update(xi, target_y)
    print("")
    return

# -----------------------------------------------------------

def main():
  print("\nBegin Tsetlin Machine binary classification ")

  print("\nLoading binary-features two-class Iris data ")
  train_file = ".\\Data\\iris_two_classes_train_80.txt"
  train_X = np.loadtxt(train_file, 
    usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    delimiter=",", comments="#", dtype=np.int32)
  print("\nFirst three X train items: ")
  print(train_X[0:3,:])
  train_y = np.loadtxt(train_file, usecols=16,
   delimiter=",", comments="#", dtype=np.int32)
  print("\nFirst three y target (0-1) values: ")
  for i in range(3):
    print(train_y[i])

  test_file = ".\\Data\\iris_two_classes_test_20.txt"
  test_X = np.loadtxt(test_file, 
    usecols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    delimiter=",", comments="#", dtype=np.int32)
  test_y = np.loadtxt(test_file, usecols=16,
    delimiter=",", comments="#", dtype=np.int32)
    
  n_clauses = 20  # number rules
  n_features = 16
  n_states = 50
  s = 3.0  # how often random updates are, using 1/s
  threshold = 10  # aka T controls voting

  print("\nSetting n_clauses = " + str(n_clauses))
  print("Setting n_states = " + str(n_states))
  print("Setting s (random update inverse " + \
    "frequency) = %0.1f " % s)
  print("Setting threshold (voting max/min) = " + \
    str(threshold))
  print("Creating Tsetlin Machine binary classifier ")
  tm = TsetlinMachine(n_clauses, n_features, n_states,
    s, threshold)
  print("Done ")

  max_epochs = 100
  print("\nSetting max_epochs = " + str(max_epochs))
  print("Starting training ")
  tm.fit(train_X, train_y, max_epochs)
  print("Done ")

  train_acc = tm.accuracy(train_X, train_y)
  print("\nAccuracy (train) = %0.4f " % train_acc)

  test_acc = tm.accuracy(test_X, test_y)
  print("Accuracy (test) = %0.4f " % test_acc)

  print("\nPrediciting class for train_X[0] ")
  pred_y = tm.predict(train_X[0])
  print("Predicted y = " + str(pred_y))

  print("\nEnd demo ")

if __name__ == "__main__":
  main()
