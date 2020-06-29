from nltk.translate.bleu_score import sentence_bleu
import sys
import csv
truth_list_csv = sys.argv[1] # data to translate
pred_list_csv = sys.argv[2] # where to put translations



def calculate_bleu(truth_list, pred_list):
    assert len(truth_list) == len(pred_list)
    score_list = []
    for i in range(len(truth_list)):
        reference = [truth_list[i]]
        candidate = pred_list[i]
        assert len(candidate) > 0 and len(reference[0]) > 0
        if len(candidate) ==1 or len(reference[0]) ==1:
            score = sentence_bleu(reference, candidate, weights=(1,0,0,0))
        elif len(candidate) ==2 or len(reference[0]) ==2:
            score = sentence_bleu(reference, candidate, weights=(0.5, 0.5,0,0))
        elif len(candidate) ==3 or len(reference[0]) ==3:
            score = sentence_bleu(reference, candidate, weights=(0.3333, 0.3333, 0.3333,0))
        else:
            score = sentence_bleu(reference, candidate)
        score_list.append(score)
    return score_list

def calculate_exact_match(truth_list, pred_list):
  match_count = 0 
  print(truth_list[0])
  print(pred_list[0])
  for i in range(len(truth_list)):
    truth = truth_list[i][2:-2]
    truth = truth.split()
    pred = pred_list[i].split()
    truth = " ".join(truth)
    pred = " ".join(pred)
    if truth == pred:
      match_count+=1
  print(truth)
  return match_count/len(truth_list)

with open(truth_list_csv) as my_file:
  csv_reader1 = csv.reader(my_file, delimiter=',')
  truth_list= []
  for row in csv_reader1:
    truth_list.append(row[1])

with open(pred_list_csv) as my_file:
  csv_reader2 = csv.reader(my_file, delimiter=',')
  pred_list= []
  for row in csv_reader2:
    pred_list.append(row[1])

bleu_score = calculate_bleu(truth_list, pred_list)
match_percent = calculate_exact_match(truth_list, pred_list)

bleu_sum=0
for i in bleu_score:
  bleu_sum+=i

print(bleu_sum/len(bleu_score),match_percent)