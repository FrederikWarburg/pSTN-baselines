######################################################################################################
# CLASSIFIER
# MNIST
parameter_dict_classifier_MNIST_CNN = {
    'nr_target_classes': 10,
    'CNN_filters1': 12,
    'CNN_filters2': 24,
    'CNN_kernel_size': 5,
    'resulting_size_classifier': 24 * 4 * 4, # default is mnist; we override for random_placement_mnist below
    'hidden_layer_classifier': 52,
    'color_channels': 1
}

parameter_dict_classifier_MNIST_STN = {
    'nr_target_classes': 10,
    'CNN_filters1': 10,
    'CNN_filters2': 20,
    'CNN_kernel_size': 5,
    'resulting_size_classifier': 20 * 4 * 4,
    'hidden_layer_classifier': 50,
    'color_channels': 1
}

parameter_dict_classifier_MNIST_P_STN = {
    'nr_target_classes': 10,
    'CNN_filters1': 10,
    'CNN_filters2': 20,
    'CNN_kernel_size': 5,
    'loc_kernel_size': 5,
    'resulting_size_classifier': 320,
    'hidden_layer_classifier': 50,
    'color_channels': 1
}

# rotated MNIST = MNIST for classifier


# random placement MNIST
parameter_dict_classifier_RandomPlacementMNIST_CNN = {
    'nr_target_classes': 10,
    'CNN_filters1': 12,
    'CNN_filters2': 20,
    'CNN_kernel_size': 5,
    'resulting_size_classifier': 20 * 21 * 21, 
    'hidden_layer_classifier': 52,
    'color_channels': 1
}


parameter_dict_classifier_RandomPlacementMNIST_STN = {
    'nr_target_classes': 10,
    'CNN_filters1': 10,
    'CNN_filters2': 20,
    'CNN_kernel_size': 5,
    'resulting_size_classifier': 20 * 21 * 21,
    'hidden_layer_classifier': 30,
    'color_channels': 1
}


parameter_dict_classifier_RandomPlacementMNIST_P_STN = {
    'nr_target_classes': 10,
    'CNN_filters1': 10,
    'CNN_filters2': 20,
    'CNN_kernel_size': 5,
    'loc_kernel_size': 5,
    'resulting_size_classifier':  20 * 21 * 21,
    'hidden_layer_classifier': 30,
    'color_channels': 1
}


##########################################################################################################################
# LOCALISER
# smaller models for regular MNIST experiments 
parameter_dict_localiser_MNIST_STN  = {
    'loc_kernel_size': 5,
    'resulting_size_localizer': 18 * 4 * 4,
    'max_pool_res': 2,
    'hidden_layer_localizer': 50,
    'localizer_filters1': 12,
    'localizer_filters2': 18,
    'color_channels': 1
}

parameter_dict_localiser_MNIST_P_STN = {
    'loc_kernel_size': 5,
    'resulting_size_localizer': 14 * 4 * 4,
    'max_pool_res': 2,
    'hidden_layer_localizer': 38,
    'localizer_filters1': 8,
    'localizer_filters2': 14,
    'color_channels': 1
}

# larger models for rotMNIST + frozen classifier experiments 
parameter_dict_localiser_rotMNIST_STN  = {
    'loc_kernel_size': 5,
    'resulting_size_localizer': 64 * 4 * 4,
    'max_pool_res': 2,
    'hidden_layer_localizer': 20,
    'localizer_filters1': 32,
    'localizer_filters2': 64,
    'localizer_filters3': 128,
    'localizer_filters4': 256,
    'color_channels': 1
}

parameter_dict_localiser_rotMNIST_P_STN  = {
    'loc_kernel_size': 5,
    'resulting_size_localizer': 53 * 4 * 4, 
    'max_pool_res': 2,
    'hidden_layer_localizer': 20,
    'localizer_filters1': 28,
    'localizer_filters2': 53,
    'localizer_filters3': 106,
    'localizer_filters4': 212,
    'color_channels': 1
}

# even larger for random placement MNIST
parameter_dict_localiser_RandomPlacementMNIST_STN = {
    'loc_kernel_size': 7,
    'resulting_size_localizer': 31 * 19 * 19,
    'max_pool_res': 2,
    'hidden_layer_localizer': 15,
    'localizer_filters1': 16,
    'localizer_filters2': 31,
    'localizer_filters3': 64,
    'localizer_filters4': 128,
    'color_channels': 1
}

parameter_dict_localiser_RandomPlacementMNIST_P_STN = {
    'loc_kernel_size': 7,
    'resulting_size_localizer': 21 * 19 * 19,
    'max_pool_res': 2,
    'hidden_layer_localizer': 12,
    'localizer_filters1': 11,
    'localizer_filters2': 21,
    'localizer_filters3': 42,
    'localizer_filters4': 84,
    'color_channels': 1
}


