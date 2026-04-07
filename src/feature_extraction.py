import numpy as np

class FeatureExtractor:
    def __init__(self, threshold_percentile=90):
        '''
        threshold_percentile: threshold to identify 'high stress' regions
        '''
        self.threshold_percentile = threshold_percentile

    def extract_spatial_features(self, von_mises_stress_map):
        '''
        Extracts spatial features from a single stress map.
        features:
        - max_stress
        - mean_stress
        - std_stress
        - high_stress_area_ratio (percentage of area exceeding the threshold percentile)
        - max_stress_gradient
        '''
        max_stress = np.max(von_mises_stress_map)
        mean_stress = np.mean(von_mises_stress_map)
        std_stress = np.std(von_mises_stress_map)
        
        # High stress area
        threshold_val = np.percentile(von_mises_stress_map, self.threshold_percentile)
        high_stress_area = np.sum(von_mises_stress_map >= threshold_val)
        total_area = von_mises_stress_map.size
        high_stress_ratio = high_stress_area / total_area

        # Gradients
        dy, dx = np.gradient(von_mises_stress_map)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        max_grad = np.max(gradient_magnitude)

        return np.array([max_stress, mean_stress, std_stress, high_stress_ratio, max_grad])

    def extract_temporal_features(self, stress_map_sequence):
        '''
        Extracts temporal features from a sequence of stress maps.
        '''
        spatial_features = [self.extract_spatial_features(sm) for sm in stress_map_sequence]
        spatial_features = np.array(spatial_features) # shape (T, num_features)
        
        # Aggregation over time
        mean_over_time = np.mean(spatial_features, axis=0)
        max_over_time = np.max(spatial_features, axis=0)
        variance_over_time = np.var(spatial_features, axis=0)
        
        # Combine temporal aggregations into a single feature vector
        combined_features = np.concatenate([mean_over_time, max_over_time, variance_over_time])
        return combined_features
