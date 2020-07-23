import torch
from time import time
from dataset import PlanetDataset
from torch.utils.data import DataLoader
from model import PlanetModel
from data_evaluation import weird_mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 1000
BS = 500
PRINT_INTERVAL = 10
INPUT = 'matrix'

ENSEMBLE_SIZE = 5
FOLDS = 5


def wmae_loss(predictions, reals):
    return torch.mean(reals * torch.nn.functional.l1_loss(predictions, reals, reduction='none'))


errors = []
for TEST_FOLD in range(FOLDS):
    print()
    print('#'*50)
    print('Starting fold {}'.format(TEST_FOLD))

    print('Loading the data...')
    train_folds = [[i for i in range(FOLDS) if i != TEST_FOLD] for j in range(ENSEMBLE_SIZE)]
    # train_folds = [[i for i in range(FOLDS)] for j in range(ENSEMBLE_SIZE)]
    trainsets = [PlanetDataset(FOLDS, train_folds[i]) for i in range(ENSEMBLE_SIZE)]
    testset = PlanetDataset(FOLDS, [TEST_FOLD])
    prediction_set = PlanetDataset(1, [0], train_or_test='test')
    train_loaders = [DataLoader(trainset, shuffle=True, batch_size=BS) for trainset in trainsets]
    test_loader = DataLoader(testset, batch_size=BS)
    prediction_loader = DataLoader(prediction_set, batch_size=BS)
    print('Done!')

    models = [PlanetModel().to(DEVICE).float() for _ in range(ENSEMBLE_SIZE)]
    loss = torch.nn.MSELoss()

    optimizers = [torch.optim.Adam(model.parameters()) for model in models]


    def evaluate(loader, error=True):
        for model in models:
            model.eval()
        reals = []
        predictions = []
        planets = []
        with torch.no_grad():
            for batch in loader:
                if error:
                    reals.append(batch['radii'])
                planets.extend(batch['planet'])
                matrix = batch[INPUT].to(DEVICE).float()
                radiation = batch['radiation'].to(DEVICE).float()
                misc = batch['misc_inputs'].to(DEVICE).float()
                maxes = batch['maxes'].to(DEVICE).float()
                means = batch['relative_means'].to(DEVICE).float()
                # tsfresh = batch['tsfresh'].to(DEVICE).float()
                others = torch.cat([radiation, misc, maxes, means], dim=1)
                preds = [model(matrix, others)[0].cpu().numpy() for model in models]
                pred = np.mean(preds, axis=0)
                predictions.append(pred)

        for model in models:
            model.train()
        predictions = np.concatenate(predictions)
        if error:
            reals = np.concatenate(reals)
            mse = weird_mean_absolute_error(predictions, reals)
        else:
            mse = 0
        return planets, predictions, reals, mse


    radii_estimate = 0
    sma_estimate = 0
    incl_estimate = 0
    loss_estimate = 0
    start_time = time()
    for epoch in range(1, EPOCHS+1):
        # if epoch == 500:
        #     optimizers = [torch.optim.Adam(model.parameters(), lr=1e-4) for model in models]
        if epoch == 750:
            optimizers = [torch.optim.Adam(model.parameters(), lr=1e-6) for model in models]

        for i in range(ENSEMBLE_SIZE):
            for batch in train_loaders[i]:
                matrix = batch[INPUT].to(DEVICE).float()
                radiation = batch['radiation'].to(DEVICE).float()
                misc = batch['misc_inputs'].to(DEVICE).float()
                maxes = batch['maxes'].to(DEVICE).float()
                means = batch['relative_means'].to(DEVICE).float()
                # tsfresh = batch['tsfresh'].to(DEVICE).float()
                others = torch.cat([radiation, misc, maxes, means], dim=1)
                radii = batch['radii'].to(DEVICE).float()
                sma = 10**-10*batch['sma'].to(DEVICE).float()
                incl = torch.abs(batch['incl'] - 90).to(DEVICE).float()

                pred_radii, pred_sma, pred_incl = models[i](matrix, others)
                loss_radii = loss(pred_radii, radii)
                loss_sma = loss(pred_sma, sma)
                loss_incl = loss(pred_incl, incl)
                total_loss = loss_radii + 0.001*loss_sma + 0.00001*loss_incl
                total_loss.backward()

                radii_estimate += loss_radii.item()
                sma_estimate += loss_sma.item()
                incl_estimate += loss_incl.item()
                loss_estimate += total_loss.item()
                optimizers[i].step()
                models[i].zero_grad()

        if epoch % PRINT_INTERVAL == 0:
            _, _, _, e = evaluate(test_loader)
            print('  '.join([
                'Time: {:<10}'
                'Epoch: {:<5}',
                'radii: {:8.6f}',
                'sma: {:8.6f}',
                'incl: {:8.6f}',
                'Total: {:8.6f}',
                'Test wmae: {:8.6f}',
            ]).format(
                (time() - start_time) // 1,
                epoch,
                radii_estimate / PRINT_INTERVAL,
                sma_estimate / PRINT_INTERVAL,
                incl_estimate / PRINT_INTERVAL,
                loss_estimate / PRINT_INTERVAL,
                e
            ))
            radii_estimate = 0
            sma_estimate = 0
            incl_estimate = 0
            loss_estimate = 0

    planets, predictions, reals, e = evaluate(test_loader)
    print('Fold {} final test error: {}'.format(TEST_FOLD, e))
    errors.append(e)
    # with open('inspection{}.txt'.format(TEST_FOLD), 'w') as f:
    #     print('planet,tag,radii', file=f)
    #     for i in range(len(planets)):
    #         print('{},real,{}'.format(planets[i], list(reals[i])), file=f)
    #         print('{},pred,{}'.format(planets[i], list(predictions[i])), file=f)

    # channels = [0, 1, 10, 43, 54]
    # f, axarr = plt.subplots(1, len(channels))
    # indices = list(range(reals.shape[0]))
    # indices.sort(key=lambda x: np.linalg.norm(reals[x]))
    # for i in range(len(channels)):
    #     axarr[i].set_title('Channel {}'.format(channels[i]))
    #     axarr[i].plot([reals[j][channels[i]] for j in indices], 'o', ms=3)
    #     axarr[i].plot([predictions[j][channels[i]] for j in indices], 'o', ms=3)
    # f.subplots_adjust(hspace=0.3)
    # plt.show()

    planets, predictions, _, _ = evaluate(prediction_loader, error=False)
    with open('predictions.txt', 'w') as f:
        print('planet,radii', file=f)
        for i in range(len(planets)):
            print('{},{}'.format(planets[i], list(predictions[i])), file=f)

print('{}-fold cv error: {}'.format(FOLDS, np.mean(errors)))
