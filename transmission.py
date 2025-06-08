import numpy as np
import matplotlib.pyplot as plt

# Constantes fondamentales (unités arbitraires)
hbar = 1.0  # Constante de Planck réduite
m = 1.0     # Masse de la particule

# Paramètres du puits
V0 = 10.0   # Profondeur du puits (V = -V0)
a = 1.0     # Largeur du puits

# Énergies (éviter E=0)
E = np.linspace(0.1, 50, 1000)

# Calculs
argument = (2 * a / hbar) * np.sqrt(2 * m * (E + V0))
sin2_term = np.sin(argument) ** 2
T_inv = 1 + (V0**2) / (4 * E * (E + V0)) * sin2_term

# Transmission et réflexion
T = 1 / T_inv
R = 1 - T

# Tracé
plt.figure(figsize=(10, 6))
plt.plot(E, T, label="Transmission T(E)", color='blue')
plt.plot(E, R, label="Réflexion R(E)", color='orange')
plt.axhline(1, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel("Énergie E")
plt.ylabel("Coefficient")
plt.title("Transmission et Réflexion – Effet Ramsauer-Townsend")
plt.legend()
plt.grid(True)
plt.ylim(0, 1.1)
plt.show()
