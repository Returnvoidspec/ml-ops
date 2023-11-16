import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os

class MyHandler(FileSystemEventHandler):
    def __init__(self, threshold):
        self.threshold = threshold
        self.added_files_count = 0

    def on_created(self, event):
        if event.is_directory:
            return

        if event.src_path.lower().endswith(('.jpg', '.jpeg')):
            self.added_files_count += 1
            print(f"Fichier JPEG ajouté: {event.src_path}")

            if self.added_files_count >= self.threshold:
                print(f"Seuil atteint. Déclenchement du réentraînement.")
                # Exécuter le script de réentraînement
                os.system("python training.py")
                # Réinitialiser le compteur
                self.added_files_count = 0

# Chemin à surveiller
train_path = "chemin/vers/train"
# Nombre de fichiers à ajouter avant de déclencher le réentraînement
threshold = 10  # Remplacez 10 par le seuil désiré

if __name__ == "__main__":
    event_handler = MyHandler(threshold)
    observer = Observer()
    observer.schedule(event_handler, train_path, recursive=True)

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
