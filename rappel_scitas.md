# 🖥️ Connexion SCITAS — Rappel

## Chaque fois que tu veux travailler

### 1. Se connecter au cluster
```powershell
ssh garate@izar.epfl.ch
```

### 2. Lancer un job GPU interactif
```bash
srun -t 120 -A cs-503 --qos=cs-503 --gres=gpu:1 --mem=16G --pty bash
```
> ⏱️ Durée max : 120 minutes. Relance si expiré.

### 3. Vérifier le noeud GPU
```bash
hostname
# ex: i03
```

### 4. Activer l'environnement
```bash
conda activate nanofm
```

### 5. Connecter VS Code au noeud GPU
- `Ctrl+Shift+P` → `Remote-SSH: Connect to Host` → `izar-gpu`
- Ouvrir le dossier : `/home/garate/Visual Intelligence/NanoFM_Homeworks`
- Sélectionner le kernel : **Jupyter Kernel... → nanofm**

---

## ⚠️ Si le noeud a changé (ex: i03 → i05)

Mettre à jour `C:\Users\tjga9\.ssh\config` :
```
Host izar-gpu
  HostName i05   # ← changer ici
  User garate
  ProxyJump izar
```

Et recopier la clé SSH sur le nouveau noeud depuis le terminal SSH sur le cluster:
```bash
echo "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAH8gOxE/UTV7z/IHpta8f5q/Fi/Q0V9UKJYpua5fViS tjga.98@gmail.com" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

---

## 📁 Chemins utiles
| Quoi | Où |
|---|---|
| Code | `/home/garate/Visual Intelligence/NanoFM_Homeworks` |
| Datasets/checkpoints | `/scratch/garate/` |

---

## 🚀 Lancer un entraînement (batch)
```bash
cd ~/Visual\ Intelligence/NanoFM_Homeworks
sbatch submit_job.sh cfgs/nanoGPT/tinystories_d8w512.yaml <wandb_key> 1
```

Vérifier le job :
```bash
squeue -u garate
```

Annuler un job :
```bash
scancel <job-id>
```
