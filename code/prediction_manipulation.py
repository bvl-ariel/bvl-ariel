import numpy as np
import os
from typing import Iterable
from data_evaluation import wmae
import random


def make_official_predictions(file_in):
    file_out = file_in + ".ccssvv"
    with open(file_out, "w") as f:
        with open(file_in) as g:
            for l in g:
                line = l.strip()
                for _ in range(100):
                    print(line, file=f)


def prepare_submission(prediction_file):
    with open(prediction_file) as f:
        f.readline()
        predictions = f.readlines()

    predictions.sort()

    with open('submission.csv', 'w') as f:
        for prediction in predictions:
            sub = prediction.strip()[6:].replace(',', '').replace(']', '').replace(' ', '\t')
            print(sub, file=f)


def clus_submission():
    predictions = {}
    with open('clus/settings.ens.test.preds') as f:
        in_data = False
        for line in f:
            if line.strip().lower() == '@data':
                in_data = True
            elif line.startswith('% Target'):
                continue
            elif in_data:
                words = line.strip().split(',')
                if len(words) > 3:
                    planet = int(words[0])
                    if planet not in predictions:
                        predictions[planet] = [0] * 55
                    channel = int(words[1])
                    predictions[planet][channel] = float(words[3])

    planets = sorted(list(predictions.keys()))
    with open('clus_submission.txt', 'w') as f:
        for planet in planets:
            for _ in range(100):
                print('\t'.join([str(x) for x in predictions[planet]]), file=f)


def ensemble_of_predictions(files, weights, experiment_dir):
    def weighted_predictions(ws, d):
        return np.sum(w * ps for w, ps in zip(ws, d))

    def convex_combination(ws, d):
        return np.average(d, axis=0, weights=ws)

    predictions = np.array([np.genfromtxt(file, delimiter='\t') for file in files])
    ensemble = np.zeros((629, 55))

    print("Merging", files)
    if not isinstance(weights[0], Iterable):
        print("Assuming that all channels are aggregated in the same way.")
        weights = [weights for _ in range(55)]

    for channel in range(55):
        data = predictions[:, :, channel]
        ensemble[:, channel] = weighted_predictions(weights[channel], data)
    # ensemble = np.average(predictions, axis=0, weights=weights)
    os.makedirs(experiment_dir, exist_ok=True)
    np.savetxt(os.path.join(experiment_dir, "predictions_test.csv"), ensemble, delimiter='\t',
               fmt='%.18f')
    with open(os.path.join(experiment_dir, "meta.txt"), "w") as f:
        print(files, file=f)
        print(weights, file=f)


def find_optimal_weights_channel(prediction_files, true_values, channel, w0):
    def current_predictions(ws):
        return np.sum(w * ps for w, ps in zip(ws, predictions))

    # def gradient():
    #     def sign(t):
    #         return 1.0 if t >= 0 else -1.0
    #
    #     g = []
    #     for ps in predictions:
    #         signs = np.array([sign(p - y) for p, y in zip(ps, ys)])
    #         s = np.sum(ys * ps * signs)
    #         g.append(s)
    #     return np.array(g)

    print("Optimising channel", channel)
    predictions = np.array([np.genfromtxt(file, delimiter='\t')[:, channel] for file in prediction_files])
    ys = np.genfromtxt(true_values, delimiter='\t')[:, channel]

    best_error = float("inf")
    best_solution = None
    if w0 is None:
        starting_points = [
            [1/3 for _ in range(3)]] + [[i / 10, j / 10, (10 - i - j) / 10] for i in range(11) for j in range(11 - i)]
        # starting_points = []
        random.seed(123)
        for _ in range(0):
            t = random.random()
            u = random.random() * (1 - t)
            starting_points.append([t, u, 1 - t - u])
    else:
        starting_points = [w0]
    all_solutions = []
    for w0 in starting_points[::-1]:
        error_current = float("inf")
        w_current = np.array(w0)
        print("  testing", w0)
        for i in range(1, 1001):
            # print("Iteration", i)
            ys_hat = current_predictions(w_current)
            error_current = wmae(ys, ys_hat)
            # print("Current w and error:", w_current, error_current)
            # delta = gradient()

            # print("Gradient:", delta)
            found_better = False
            w_new = [w for w in w_current]
            dimension = i % len(w0)
            for sign in [1, -1]:
                if not found_better:
                    w_new = [w for w in w_current]
                step = 0.01
                while True:
                    w_new[dimension] = w_current[dimension] + sign * step
                    ys_hat = current_predictions(w_new)
                    error_new = wmae(ys, ys_hat)
                    if error_new < error_current:
                        found_better = True
                        break
                    step /= 2.0
                    if step < 10 ** -10:
                        break
                if found_better:
                    break
            # print("Step", step)
            w_current = w_new
            # if min(w_current) <= 0:
            #     print("----> ", w_current)
        # print("---->", w_current)
        all_solutions.append(error_current)
        if error_current < best_error:
            best_error = error_current
            best_solution = w_current
            print("    new best solution", best_solution, best_error)
    print("Mean and Variance of the solutions:", np.mean(all_solutions), np.std(all_solutions))
    return best_solution


def find_optimal_weights(prediction_files, true_values, channels, ws0):
    return [find_optimal_weights_channel(prediction_files, true_values, c, ws0[c]) for c in channels]


def ensemble_winners_per_target(files: Iterable[str], wmae_performance: np.ndarray, out_dir):
    """
    :param files: paths to non-inflated csv files with predictions
    :param wmae_performance: wmae_performance[i][j] = performance of model i for channel j
    :return:
    """
    best_models = np.argmax(wmae_performance, axis=0)
    predictions = [open(f) for f in files]
    with open(os.path.join(out_dir, "predictions_test.csv"), "w") as f:
        for lines in zip(*predictions):
            new_line = [""] * 55
            for i, best in enumerate(best_models):
                new_line[i] = str(lines[best][i])
            print('\t'.join(new_line), file=f)

    for f in predictions:
        f.close()


def find_indices(id_file):
    with open(id_file) as f:
        ids = [x.strip() for x in f.readlines()]
    ids_all = []
    previous = ""
    with open("../database/noisy_train.txt") as f:
        for l in f:
            after_slash = l[l.find("/") + 1:]
            i = after_slash[: after_slash.find("_")]
            assert len(i) == 4
            if i != previous:
                ids_all.append(i)
            previous = i
    id_to_index = {ii: i for i, ii in enumerate(ids_all)}
    return [id_to_index[ii] for ii in ids]


if __name__ == "__main__":
    # prepare_submission('predictions.txt')
    # make_official_predictions('submission.csv')

    # make_official_predictions("../experiments/xgb_cluster_optimised_str_optimised_wmae/predictions_test.csv")

    # ensemble_of_predictions(["../experiments/submission15_rf_xgb_cnn/predictions_test.csv",
    #                          "../experiments/submission13_ensemble_cnn-pct-xgb/predictions_test.csv"],
    #                         [0.5, 0.5],
    #                         "../experiments/submission16_ensemble_ensemble")
    # make_official_predictions("../experiments/submission16_ensemble_ensemble/predictions_test.csv")

    if 0:
        ensemble_of_predictions(["../experiments/bagging_latest_feat_optimised2/predictions_test.csv",
                                 "../experiments/bagging_of_200boosting/predictions_test.csv",
                                 "../experiments/submission21_fcnn_ensemble/predictions_test.csv"],
                                [0.15, 0.25, 0.6],
                                "../experiments/submission44")
        make_official_predictions("../experiments/submission44/predictions_test.csv")
    if 0:
        ensemble_of_predictions(["../experiments/bagging_latest_feat_optimised2/predictions_test.csv",
                                 "../experiments/bagging_of_200boosting/predictions_test.csv"],
                                [0.5, 0.5],
                                "../experiments/submission44")
        # make_official_predictions("../experiments/submission44/predictions_test.csv")
    if 0:
        ensemble_of_predictions(["../experiments/bagging_latest_feat_optimised/predictions_test.csv",
                                 "../experiments/submission27_semi_mtr_bagg250/predictions_test.csv",
                                 "../experiments/submission28_weird_mae_bagging_250/predictions_test.csv",
                                 "../experiments/submission10_bagging_str_500_optimised/predictions_test.csv"],
                                [0.25, 0.25, 0.25, 0.25], "../experiments/aggregated_bagging")
    make_official_predictions("../experiments/submission44/predictions_test.csv")


    # clus_submission()
    if 0:
        exp_dir = "../optimisation/best_w/"
        # filtered_indices = find_indices(os.path.join(exp_dir, "planets.txt"))
        a = find_optimal_weights([os.path.join(exp_dir, "RandomForest.csv"),
                                  os.path.join(exp_dir, "GradientBoosting.csv"),
                                  os.path.join(exp_dir, "cnn.csv")],
                                 os.path.join(exp_dir, "true_values.csv"), list(range(35, 36)), [None for _ in range(55)])
        print(a)
    if 0:
        ensemble_of_predictions(["../experiments/submission10_bagging_str_500_optimised/predictions_test.csv",
                                 "../experiments/submission11_xgb_cluster_optimised_str_optimised_wmae/predictions_test.csv",
                                 "../experiments/best_cnn_submission/pred_cnn.csv"],
                                [[0.2667976379394517, 0.1295774054527279, 0.6098425458371626],
                                 [0.23066406249999893, 0.13401885986328105, 0.6411816406250003],
                                 [0.16781314283609258, 0.17335811868310075, 0.6640897567570205],
                                 [0.20119157433509907, 0.20465784505009615, 0.5976538316905543],
                                 [0.200583407431842, 0.20468695476651091, 0.5976586517691683],
                                 [0.20124842166900678, 0.20283144474029516, 0.598495221287012],
                                 [0.3002630603313446, 0.10496566832065583, 0.5975170846283435],
                                 [0.20435629263520227, 0.1982305544614798, 0.6000277905166157],
                                 [0.2006200771033775, 0.20492207869887233, 0.5975096857547872],
                                 [0.300270864814516, 0.1048933802545062, 0.5994234083592946],
                                 [0.30091827332973475, 0.10454086139798166, 0.5978492006659513],
                                 [0.3011731386184692, 0.10470735460519792, 0.5976463221013545],
                                 [0.3011767674982548, 0.10441345259547236, 0.5976467894017696],
                                 [0.21272212773561502, 0.19259036988019926, 0.5974879796803005],
                                 [0.30187946751713735, 0.10627144396305088, 0.5949038600921633],
                                 [0.29632184982300097, 0.10680457696318715, 0.5998973679542552],
                                 [0.19502675056457497, 0.20663856059312768, 0.6011617529392226],
                                 [0.25888671875000036, 0.12555717468261735, 0.6181576728820756],
                                 [0.17299540206790034, 0.20723570421338092, 0.6226159010827559],
                                 [0.08737509369850166, 0.3028135398030277, 0.6126249027252193],
                                 [0.17104573264718095, 0.2111847895383836, 0.6201623916625981],
                                 [0.19619471549987752, 0.2049811947345734, 0.6010611437261111],
                                 [0.24882649078965102, 0.18136524543166166, 0.5720235484838468],
                                 [0.20046316996216945, 0.20492467999458192, 0.5976519221067461],
                                 [0.267810294479131, 0.08435995608568238, 0.651881022006274],
                                 [0.3327793242037288, 0.082630820274353, 0.5874207055568715],
                                 [0.23068115234374842, 0.1614242553710955, 0.6111816406250007],
                                 [0.23411540031433037, 0.18234846889972742, 0.5861911635100866],
                                 [0.30113705426454557, 0.1043559290468694, 0.597937500476837],
                                 [0.19600387573242184, 0.21255142211914121, 0.5945338988304154],
                                 [0.20062372073531262, 0.20468820482492245, 0.5981261008977994],
                                 [0.19039351254701728, 0.2064493364095678, 0.6070286351442362],
                                 [0.1006298930943012, 0.30484130710363383, 0.5974951551854613],
                                 [0.15566649660468385, 0.20121191501617294, 0.6472925658524012],
                                 [0.11389022827148457, 0.10303359985351541, 0.7879929345846188],
                                 [0.050051940381527174, 0.08497402980923756, 0.8722226481139758],
                                 [-0.12029706726471605, 0.6251896773775405, 0.4976688269277418],
                                 [0.00041901066899299695, 0.20970050752162947, 0.7963387955725192],
                                 [0.12180745363235491, 0.12833692297339483, 0.7548019272089029],
                                 [0.30824191033840165, 0.10075651675462755, 0.5931681692600275],
                                 [0.19178455397486677, 0.20785783991217588, 0.6029938749969024],
                                 [0.16023132324218772, 0.21232391357421834, 0.6311828613281236],
                                 [0.300400390625, 0.064424438476563, 0.6387622070312509],
                                 [0.30128541946411136, 0.1047125291824341, 0.5976587302982806],
                                 [0.30163451761007226, 0.10604932680726084, 0.5950152701139468],
                                 [0.3064875744283169, 0.0995773462951187, 0.5971477852761762],
                                 [0.30135303124785484, 0.10434427231550221, 0.5978389184176933],
                                 [0.30107175305485456, 0.10436650961637371, 0.5978265385329798],
                                 [0.29998221129178715, 0.053156073093414746, 0.6506804136931875],
                                 [0.20279920041561095, 0.20469470918178578, 0.5959117427468311],
                                 [0.30032162487506897, 0.10483920514583517, 0.5987913510203483],
                                 [0.30078857421874117, 0.10480712890624964, 0.5976132011413581],
                                 [0.303034971207381, 0.2969815847277637, 0.3989780245721332],
                                 [0.10949056163430182, 0.37556741878390143, 0.5161654222011549],
                                 [0.09366192594170514, 0.3694401121139493, 0.5386937820911399]],
                                "../experiments/submission15_rf_xgb_cnn"
                                )
    if 0:
        make_official_predictions("../experiments/semi_mtr_bagg250/predictions_test.csv")
    # if 0:
    #     exp_dir = "../optimisation/best_w/"
    #     filtered_indices = find_indices(os.path.join(exp_dir, "planets.txt"))
    #     a = find_optimal_weights([os.path.join(exp_dir, "RandomForest.csv"),
    #                           os.path.join(exp_dir, "GradientBoosting.csv"),
    #                           os.path.join(exp_dir, "cnn.csv")],
    #                          os.path.join(exp_dir, "true_values.csv"), list(range(55)), [None for _ in range(55)])
    #     print(a)
