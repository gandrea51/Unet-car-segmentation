import torch

def early_stopping(model: torch.nn.Module, model_path: str, val_loss: float, best_val_loss: float, patience_count: int, patience: int):
    '''Funzione di Early Stopping'''

    stop = False

    if val_loss < best_val_loss:
        
        # Miglioramento: Aggiornamento e salvataggio
        best_val_loss = val_loss
        patience_count = 0
        torch.save(model.state_dict(), model_path)
        print(f'Attenzione - modello migliore salvato in {model_path}.')
    else:

        # Peggioramento / stallo: Incremento della pazienza
        patience_count += 1
        print(f'Early stopping patience: {patience_count} / {patience}.')

        if patience_count >= patience:
            
            # Pazienza raggiunta
            print(f'Early stopping triggered.')
            stop = True
    
    return best_val_loss, patience_count, stop

