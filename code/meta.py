import numpy as np
import os


pattern = r"C:\Users\matej\Documents\clusTesti\ariel\s{}.test.pred.arff"
eps = ["01", "005", "001", "0005"]


def load_pred(file):
    lines = {"any": [], "bagging": [], "fcnn": []}
    with open(file) as f:
        for x in f:
            if x.startswith("@DATA"):
                break
        for x in f:
            line = x.strip().split(',')
            pred = line[2]
            lines[pred].append(line[0])
    return lines


def see_more(file):
    lines = []
    with open(file) as f:
        for x in f:
            if x.startswith("@DATA"):
                break
        for x in f:
            line = x.strip().split(',')
            ps = [float(t) for t in line[3:6]]
            lines.append([line[0], line[2], ps])
    not_any = [line for line in lines if line[1] != "any"]
    not_any.sort(key=lambda l: max(l[2]), reverse=True)
    for l in not_any:
        print("{} {} {}a".format(l[0], l[1], max(l[2])))


def compare():
    files = [pattern.format(e) for e in eps]
    predictions = [load_pred(f) for f in files]
    bagging = set(x for prediction in predictions for x in prediction["bagging"])
    fcnn = set(x for prediction in predictions for x in prediction["fcnn"])
    print(bagging & fcnn)


def merge_predictions(out_dir, planets):
    meta_predictions = load_pred(pattern.format("0005"))
    meta_predictions = [(planet, pred) for pred in meta_predictions for planet in meta_predictions[pred]]
    meta_predictions.sort()

    predictions_any = np.genfromtxt(os.path.join(out_dir, "predictions_test_any.csv"), delimiter='\t')
    predictions_bag = np.genfromtxt(os.path.join(out_dir, "predictions_test_bag_xgb.csv"), delimiter='\t')
    # predictions_fcnn = np.genfromtxt(os.path.join(out_dir, "predictions_test_fcnn.csv"), delimiter='\t')

    total = np.zeros((629, 55))
    assert 629 == len(meta_predictions)
    for i in range(len(meta_predictions)):
        planet, _ = meta_predictions[i]
        # if prediction == "any":
        #     total[i] = predictions_any[i]
        # elif prediction == "bagging":
        #     total[i] = predictions_bag[i]
        # elif prediction == "fcnn":
        #     total[i] = predictions_fcnn[i]
        if planet in planets:
            print("correcting planet", planet)
            total[i] = predictions_bag[i]
        elif planet not in planets:
            total[i] = predictions_any[i]
        else:
            raise ValueError("!")
    np.savetxt(os.path.join(out_dir, "predictions_test.csv"), total, delimiter='\t', fmt='%.18f')


merge_predictions("../experiments/submission44", ["0611", "0855", "1097", "1692"])
# see_more(pattern.format("01"))