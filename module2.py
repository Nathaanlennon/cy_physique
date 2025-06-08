import numpy as np
import matplotlib.pyplot as plt

# Constantes physiques
ħ = 1        # Constante de Planck réduite (unités naturelles)
m = 1        # Masse réduite (unités naturelles)

# Paramètres du puits
a = 2.0      # Largeur du puits : de x = 0 à x = a
V0 = -50.0   # Profondeur du puits (doit être négatif)

# Discrétisation de l'espace
dx = 0.01
x = np.arange(-a, 2*a, dx)  # Espace autour du puits
nx = len(x)

# Potentiel : V = V0 entre 0 et a, 0 ailleurs
V = np.zeros(nx)
V[(x >= 0) & (x <= a)] = V0

# Matrice Hamiltonienne (méthode des différences finies)
diag = np.full(nx, -2.0)
off_diag = np.ones(nx - 1)
H = -(ħ**2)/(2*m*dx**2) * (
        np.diag(diag) +
        np.diag(off_diag, 1) +
        np.diag(off_diag, -1)
    ) + np.diag(V)

# Diagonalisation de l'Hamiltonien
E, psi = np.linalg.eigh(H)

# Extraction des états liés (E < 0)
bound_indices = np.where(E < 0)[0]
E_bound = E[bound_indices]
psi_bound = psi[:, bound_indices]

# Normalisation des fonctions d'onde
for i in range(len(E_bound)):
    norm = np.sqrt(np.sum(np.abs(psi_bound[:, i])**2) * dx)
    psi_bound[:, i] /= norm

# Affichage
plt.figure(figsize=(10, 6))
plt.plot(x, V, 'k--', label="Potentiel (V)")

for i in range(len(E_bound)):
    plt.plot(x, psi_bound[:, i] + E_bound[i], label=f"État {i} (E = {E_bound[i]:.2f})")

plt.title("États liés dans un puits carré fini (de 0 à a)")
plt.xlabel("x")
plt.ylabel("Énergie / Fonction d'onde")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()