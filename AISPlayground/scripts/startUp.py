import os 
from datetime import datetime

class StartUp:
    # Functions:
    # Model folders
    def clear_folder(currpath=None, models=None, clear_movies=False) -> dict:
        """
        Clear folder for frames or movies. If folders do not exist, function creates them.

        Args:
        - currpath (str): current path
        - models (str or list of str): folder(s) to clear
        - clear_movies (bool): if True, clear folder for movies, otherwise clear folder for frames

        Returns: dictionary with paths for frames and movies folders, keys: model, and a sub directory with keys: image, movies, values are paths to folders
        """

        if models is None:
            raise ValueError("No folder specified")

        if isinstance(models, str):
            models = [models]

        # Create result folder
        results_path = os.path.join(currpath, 'results')
        os.makedirs(results_path, exist_ok=True)

        paths = {}

        for model in models:
            model_folder = os.path.join(results_path, model)
            os.makedirs(model_folder, exist_ok=True)

            # Create paths to figure and movie folders
            figure_storage = os.path.join(model_folder, 'figures')
            video_storage = os.path.join(model_folder, 'movies')
            csv_storage = os.path.join(model_folder, 'csv')
            model_storage = os.path.join(model_folder, 'models')

            # Create folders, if they do not exist
            os.makedirs(figure_storage, exist_ok=True)
            os.makedirs(video_storage, exist_ok=True)
            os.makedirs(csv_storage, exist_ok=True)
            os.makedirs(model_storage, exist_ok=True)

            # Directory for saving frames, and cleaning it every run
            for item in os.listdir(figure_storage):
                os.remove(os.path.join(figure_storage, item))

            if clear_movies: # If movie is true, clean movies folder
                for item in os.listdir(video_storage):
                    os.remove(os.path.join(video_storage, item))
            
            paths[model] = {'images': figure_storage, 'movies': video_storage, 'csv': csv_storage, 'models': model_storage}

        return paths

    # Clear one folder
    def clear_one_folder(folder = None) -> None:
        """
        Clear one folder.

        Args:
        - folder (str): folder to clear

        Returns: None
        """
        if folder is None:
            raise ValueError("No folder specified")
        else:
            os.makedirs(folder, exist_ok=True)
            # Directory for saving frames, and cleaning it every run
            for item in os.listdir(folder):
                os.remove(os.path.join(folder, item))

    # Get time
    def get_time() -> str:
        """
        Get current time in format: YYYYMMDDHHMMSS

        Returns: string with current time
        """
        # Get current time
        current_time = datetime.now()
        # Format the time as a string
        return current_time.strftime('%Y%m%dT%H%M%S')