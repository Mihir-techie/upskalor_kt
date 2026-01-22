import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pathlib import Path

warnings.filterwarnings('ignore')


class WellnessBaselineCalculator:
    """
    A comprehensive wellness baseline calculator that processes emotional wellness data
    to establish personalized baselines for users.
    """
    
    def __init__(self):
        self.emotion_mapping = {
            'Distressed': 1, 'Exhausted': 1, 'Anxious': 1, 'Stressed': 2,
            'Tired': 2, 'Bored': 2, 'Calm': 3, 'Content': 3, 'Relaxed': 4,
            'Serene': 4, 'Happy': 5, 'Energized': 5, 'Accomplished': 5,
            'Invigorated': 5, 'Strong': 5, 'Empowered': 5, 'Fulfilled': 5,
            'Flexible': 4, 'Challenged': 3, 'Refreshed': 4, 'Understood': 4,
            'Rejuvenated': 5, 'Recharged': 4, 'Agile': 4
        }
        self.scaler = MinMaxScaler()
        
    def load_and_validate_data(self, file_path):
        """
        Load and validate the wellness dataset.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded and validated dataset
        """
        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Dataset file not found: {file_path}")
                
            df = pd.read_csv(file_path)
            
            required_columns = ['ID', 'Date', 'Primary Emotion', 'Energy Level (1-10)', 
                              'Mood After (1-10)', 'Stress Level (1-10)']
            
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, df):
        """
        Clean and preprocess the wellness data.
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        # Remove unnecessary columns
        columns_to_drop = [
            'Age', 'Gender', 'Time of Day', 'Activity Category', 'Sub-Category',
            'Activity', 'Duration (minutes)', 'Intensity', 'Secondary Emotion',
            'Mood Before (1-10)'
        ]
        
        # Only drop columns that exist
        existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(existing_cols_to_drop, axis=1)
        
        # Rename columns for consistency
        column_mapping = {
            'Mood After (1-10)': 'mind_clarity',
            'Energy Level (1-10)': 'energy_level',
            'Stress Level (1-10)': 'stress_level',
            'Primary Emotion': 'emotion_state'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Handle missing values
        df = df.dropna()
        
        return df
    
    def get_most_active_user(self, df):
        """
        Identify the user with the most data entries.
        
        Args:
            df (pd.DataFrame): Preprocessed dataset
            
        Returns:
            str: User ID with most entries
        """
        return df['ID'].value_counts().index[0]
    
    def prepare_user_data(self, df, user_id):
        """
        Prepare data for a specific user.
        
        Args:
            df (pd.DataFrame): Preprocessed dataset
            user_id (str): Target user ID
            
        Returns:
            pd.DataFrame: User-specific processed data
        """
        user_df = df[df['ID'] == user_id].copy()
        
        # Convert date and sort
        user_df['Date'] = pd.to_datetime(user_df['Date'])
        user_df = user_df.sort_values('Date').set_index('Date')
        
        # Map emotions to scores
        user_df['emotion_score'] = user_df['emotion_state'].map(self.emotion_mapping)
        
        # Drop unnecessary columns
        user_df = user_df.drop(columns=['ID', 'emotion_state'])
        
        return user_df
    
    def calculate_baseline_stats(self, user_df):
        """
        Calculate baseline statistics for wellness metrics.
        
        Args:
            user_df (pd.DataFrame): User-specific processed data
            
        Returns:
            pd.DataFrame: Baseline statistics (mean and variance)
        """
        features = ['energy_level', 'mind_clarity', 'stress_level', 'emotion_score']
        
        # Handle any missing emotion mappings
        user_df = user_df.dropna(subset=features)
        
        # Normalize features
        user_df[features] = self.scaler.fit_transform(user_df[features])
        
        # Calculate statistics
        baseline_mean = user_df[features].mean().to_frame(name='Baseline_Mean')
        baseline_variance = user_df[features].var().to_frame(name='Baseline_Variance')
        
        # Combine statistics
        baseline_stats = pd.concat([baseline_mean, baseline_variance], axis=1)
        
        return baseline_stats
    
    def generate_baseline_report(self, baseline_stats, user_id):
        """
        Generate a human-readable baseline report.
        
        Args:
            baseline_stats (pd.DataFrame): Baseline statistics
            user_id (str): User ID
            
        Returns:
            str: Formatted report
        """
        report = f"\n{'='*50}\n"
        report += f"WELLNESS BASELINE REPORT FOR USER: {user_id}\n"
        report += f"{'='*50}\n\n"
        
        for metric in baseline_stats.index:
            mean_val = baseline_stats.loc[metric, 'Baseline_Mean']
            var_val = baseline_stats.loc[metric, 'Baseline_Variance']
            
            report += f"{metric.replace('_', ' ').title()}:\n"
            report += f"  - Baseline (Normalized): {mean_val:.3f}\n"
            report += f"  - Variability: {var_val:.3f}\n\n"
        
        report += f"{'='*50}\n"
        report += "NOTE: Values are normalized (0-1 scale)\n"
        report += "Lower variability = more consistent patterns\n"
        report += f"{'='*50}\n"
        
        return report
    
    def calculate_user_baseline(self, file_path, output_file='user_baseline_stats.csv'):
        """
        Main method to calculate user baseline from dataset.
        
        Args:
            file_path (str): Path to the wellness dataset
            output_file (str): Output file for baseline statistics
            
        Returns:
            tuple: (baseline_stats, user_id, report)
        """
        print("🔄 Loading wellness data...")
        df = self.load_and_validate_data(file_path)
        
        if df is None:
            return None, None, None
        
        print("🧹 Preprocessing data...")
        df = self.preprocess_data(df)
        
        print(f"📊 Dataset info: {len(df)} records for {df['ID'].nunique()} users")
        
        user_id = self.get_most_active_user(df)
        print(f"👤 Analyzing data for user: {user_id}")
        
        user_df = self.prepare_user_data(df, user_id)
        print(f"📈 Processing {len(user_df)} records for this user")
        
        baseline_stats = self.calculate_baseline_stats(user_df)
        
        # Save to file
        baseline_stats.to_csv(output_file)
        print(f"💾 Baseline statistics saved to: {output_file}")
        
        # Generate report
        report = self.generate_baseline_report(baseline_stats, user_id)
        
        return baseline_stats, user_id, report


def main():
    """
    Main execution function for baseline calculation.
    """
    print("🚀 Starting Wellness Baseline Calculation")
    print("="*50)
    
    calculator = WellnessBaselineCalculator()
    
    dataset_file = "fitlife_emotional_dataset.csv"
    
    try:
        baseline_stats, user_id, report = calculator.calculate_user_baseline(dataset_file)
        
        if baseline_stats is not None:
            print("\n📋 BASELINE STATISTICS")
            print(baseline_stats)
            print(report)
        else:
            print("❌ Failed to calculate baseline. Please check the dataset file.")
            
    except Exception as e:
        print(f"❌ Error during baseline calculation: {e}")


if __name__ == "__main__":
    main()
