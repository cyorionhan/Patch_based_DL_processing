import torch
import torch.multiprocessing as mp
import time
import os
import Subfunction


def assemble_from_patches(num, patches, return_dict):
    image_thick = int(patches.size()[0] / (253 * 253) + 3)
    return_image = torch.zeros([image_thick, 256, 256])
    n = 0
    for i in range(image_thick - 3):
        for j in range(253):
            for k in range(253):
                return_image[i:i + 4, j:j + 4, k:k + 4] += patches[n, :].reshape(4, 4, 4)
                n += 1
    return_dict[num] = return_image


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # model basic
    model_name = 'model_c_2'
    note = '400 700 1000 1300 using as the training, using max and mean to normalize without min-max, L2 loss'
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    print('model name')
    print(model_name)
    print(note)
    process_patient_names = [2420, 2433, 2441, 2443, 2446, 2454, 2457, 2459, 2468, 2478, 2486, 2487, 2497, 2502, 2504,
                             2522, 2531, 2544, 2546, 2552, 2574, 2606, 2618, 2632, 2649, 2653, 2654, 2668, 2674, 2684,
                             2704, 2708, 2716, 2718, 2722, 2724, 2726, 2728, 2729, 2732, 2759, 2775, 2778, 2780, 2795,
                             2798, 2799, 2801, 2809, 2818, 2843, 2857, 2862, 2864, 2866, 2905, 2920, 2925, 2950, 2957,
                             2995, 2999, 3019, 3021, 3022, 3034, 3037, 3049, 3055, 3060, 3062, 3063, 3065, 3066, 3075,
                             3090, 3097, 3129, 3131, 3140, 3156, 3159, 3165, 3166, 3171, 3184, 3192, 3195]
    # ------------------------------------------------------------------
    # training parameters
    patch_size = [4, 4, 4]
    patch_length = patch_size[0] * patch_size[1] * patch_size[2]
    # Number of  patches
    normalization_factor = 100000
    image_types = ['700', 'TOFOSEMPSF']
    times_types = ['20sec']
    l_image_types = len(image_types)
    # ------------------------------------------------------------------
    # constant
    ANN_module_path = 'ANN_model/'
    image_size = [256, 256]
    device = 'cuda'

    # ------------------------------------------------------------------

    starttime = time.time()

    path = '../4_post_processing/' + model_name + '/'
    folder_bi = os.path.exists(path)
    if not folder_bi:
        os.makedirs(path)
        print('create the result folder')
    else:
        print('there is the result folder')

    for cut_time in times_types:
        # ----------------------------------------------------------------------
        # ANN build
        model = Subfunction.network_128to64()
        model.load_state_dict(torch.load('ANN_model/' + model_name + '.pth', map_location=torch.device('cpu')))
        # --------------------------------------------------------------------

        #################################################
        # testing
        i = 0
        for patient_int in process_patient_names:
            patient = str(patient_int)
            # length of the test total patches
            if not os.path.exists('../1_Patch_data/' + patient + '/20sec700_patch_444_resize.pt'):
                print(patient, 'patch dont exist')
                continue
            with torch.no_grad():
                output_position = torch.load('../1_Patch_data/' + patient + '/20sec700_patch_444_resize.pt')
            l_test_total_patch = output_position.size()[0]
            del output_position
            with torch.no_grad():
                rec_input = torch.zeros([l_test_total_patch, patch_length * l_image_types])
            for k1 in range(l_image_types):
                rec_input[:, k1 * patch_length:(k1 + 1) * patch_length] = torch.load('../1_Patch_data/' + patient
                                                                                     + '/' +
                                                                                     cut_time + image_types[k1]
                                                                                     + '_patch_444_resize.pt')
            rec_input /= normalization_factor
            with torch.no_grad():
                rec_output = model(rec_input)
            rec_output *= normalization_factor

            print('begin to assemble')
            middletime = time.time()
            print('Time', (middletime - starttime) / 60)
            image_thick = l_test_total_patch / (
                        (image_size[0] - patch_size[1] + 1) * (image_size[1] - patch_size[2] + 1))
            image_thick = int(image_thick)
            output_image = torch.zeros([image_thick + patch_size[0] - 1, image_size[0], image_size[1]])

            num_multi = 20
            batch = int(image_thick / num_multi)
            manager1 = mp.Manager()
            return_dict1 = manager1.dict()
            jobs1 = []
            for i in range(num_multi - 1):
                p = mp.Process(target=assemble_from_patches,
                               args=(i, rec_output[batch * i * 253 * 253:batch * (i + 1) * 253 * 253, :],
                                     return_dict1))
                jobs1.append(p)
                p.start()
            p = mp.Process(target=assemble_from_patches,
                           args=(num_multi - 1, rec_output[batch * (num_multi - 1) * 253 * 253:l_test_total_patch, :],
                                 return_dict1))
            jobs1.append(p)
            p.start()

            for proc in jobs1:
                proc.join()
            n = 0
            for i in range(num_multi):
                a = return_dict1[i]
                for j in range(len(return_dict1[i])):
                    output_image[n, :, :] += return_dict1[i][j, :, :]
                    n += 1
                n -= 3

            output_image = output_image / 64

            i += 1
            middletime = time.time()
            print('Time', (middletime - starttime) / 60)
            torch.save(output_image, path + patient + '_' + cut_time + '_enhance.pt')
