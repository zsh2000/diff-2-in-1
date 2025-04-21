import numpy as np
import torch
import cv2
import os
dir_name = "./ab1/"
sn_list = os.listdir(dir_name)
total = 0
cnt = 0

def angular_err(gt, pred):
    prediction_error = torch.cosine_similarity(gt, pred, dim=1)
    prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
    err = torch.acos(prediction_error) * 180.0 / 3.14
    return err


def rmse(gt, pred, stored_value, stored_samples, new_samples, splits):
    gts, preds = torch.split(gt, splits), torch.split(pred, splits)
    img_aggregated_vals = [
        torch.sqrt(((gt - pred) ** 2 + 1e-6).mean())
        for gt, pred in zip(gts, preds)
        if gt.shape[0] > 0
    ]
    update_value = cumulate_mean(
        torch.mean(torch.stack(img_aggregated_vals)),
        stored_value,
        new_samples,
        stored_samples,
    )
    return update_value


def rmse_log(gt, pred, stored_value, stored_samples, new_samples, splits):
    gts, preds = torch.split(gt, splits), torch.split(pred, splits)
    img_aggregated_vals = [
        torch.sqrt(((torch.log(gt) - torch.log(pred)) ** 2 + 1e-6).mean())
        for gt, pred in zip(gts, preds)
        if gt.shape[0] > 0
    ]
    update_value = cumulate_mean(
        torch.mean(torch.stack(img_aggregated_vals)),
        stored_value,
        new_samples,
        stored_samples,
    )
    return update_value


def abs_rel(gt, pred, stored_value, stored_samples, new_samples, splits):
    update_value = cumulate_mean(
        (torch.abs(gt - pred) / gt).mean(), stored_value, new_samples, stored_samples
    )
    return update_value


def sq_rel(gt, pred, stored_value, stored_samples, new_samples, splits):
    update_value = cumulate_mean(
        (((gt - pred) ** 2) / gt).mean(), stored_value, new_samples, stored_samples
    )
    return update_value


def log10(gt, pred, stored_value, stored_samples, new_samples, splits):
    update_value = cumulate_mean(
        torch.abs(torch.log10(pred) - torch.log10(gt)).mean(),
        stored_value,
        new_samples,
        stored_samples,
    )
    return update_value


def silog(gt, pred, stored_value, stored_samples, new_samples, splits):
    gts, preds = torch.split(gt, splits), torch.split(pred, splits)
    img_aggregated_vals = [
        100 * torch.sqrt((torch.log(pred) - torch.log(gt)).var())
        for gt, pred in zip(gts, preds)
        if gt.shape[0] > 0
    ]
    update_value = cumulate_mean(
        torch.mean(torch.stack(img_aggregated_vals)),
        stored_value,
        new_samples,
        stored_samples,
    )
    return update_value


def rmse_angular(gt, pred):
    return torch.sqrt(angular_err(gt, pred)).mean()


def mean(gt, pred, stored_value, stored_samples, new_samples, splits):
    err = angular_err(gt, pred)
    update_value = cumulate_mean(err.mean(), stored_value, new_samples, stored_samples)
    return update_value


def median(gt, pred, stored_value, stored_samples, new_samples, splits):
    err = angular_err(gt, pred)
    update_value = cumulate_mean(
        err.median(), stored_value, new_samples, stored_samples
    )
    return update_value


def a1(gt, pred):
    err = angular_err(gt, pred)
    rmse_a = torch.sqrt((err ** 2 + 1e-6).mean())
    a1_err = (err < 11.25).float().mean()
    a2_err = (err < 22.5).float().mean()
    a3_err = (err < 30).float().mean()
    return err.mean(), a1_err, a2_err, a3_err, rmse_a

def get_err(gt, pred):
    return angular_err(gt, pred).cpu().numpy()

def a2(gt, pred, stored_value, stored_samples, new_samples, splits):
    err = angular_err(gt, pred)
    update_value = cumulate_mean(
        (err < 7.5).float().mean(), stored_value, new_samples, stored_samples
    )
    return update_value


def a3(gt, pred, stored_value, stored_samples, new_samples, splits):
    err = angular_err(gt, pred)
    update_value = cumulate_mean(
        (err < 11.5).float().mean(), stored_value, new_samples, stored_samples
    )
    return update_value


def a4(gt, pred, stored_value, stored_samples, new_samples, splits):
    err = angular_err(gt, pred)
    update_value = cumulate_mean(
        (err < 22.5).float().mean(), stored_value, new_samples, stored_samples
    )
    return update_value


def a5(gt, pred, stored_value, stored_samples, new_samples, splits):
    err = angular_err(gt, pred)
    update_value = cumulate_mean(
        (err < 30.0).float().mean(), stored_value, new_samples, stored_samples
    )
    return update_value


rmse_err1 = 0
rmse_err2 = 0
rmse_err3 = 0
rmse_a_total  =0
error_list = []
gt_list = sorted(os.listdir("./nyu/test/norm/"))
print(gt_list)
for i in range(1, 1+len(sn_list)//2):
    sn_img = torch.from_numpy(cv2.imread(os.path.join("./nyu/test/norm/" + gt_list[i-1])) / 255.)
    sn_img = torch.reshape(sn_img, (-1, 3))
    mask = torch.squeeze(torch.argwhere(torch.sum(sn_img, 1, keepdim=False)!=0))
    sn_img_filtered = sn_img[mask]
    
    sn_pred = torch.from_numpy(cv2.resize(cv2.imread(os.path.join(dir_name, "sn_pred_"+str(i)+".png")), (640, 480)) / 255.)
    sn_pred = torch.reshape(sn_pred, (-1, 3))
    sn_pred_filtered = sn_pred[mask]

    err_single = get_err(sn_img_filtered*2-1, sn_pred_filtered[:, [2, 1, 0]]*2-1)
    print(err_single)
    if error_list is None:
        error_list = err_single.copy()
    else:
        error_list = np.concatenate((error_list, err_single))


print('Mean %f, Median %f, Rmse %f, delta1 %f, delta2 %f delta3 %f'%(np.average(error_list), np.median(error_list), np.sqrt(np.sum(error_list * error_list)/error_list.shape),\
                    np.sum(error_list < 11.25) / error_list.shape[0],np.sum(error_list < 22.5) / error_list.shape[0],np.sum(error_list < 30) / error_list.shape[0]))

#
#
