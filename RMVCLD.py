from scipy.linalg import solve_sylvester
import numpy as np
from scipy.spatial import distance


def RMVCLD(X_list, lambda1, lambda2, lambda3, alpha, m, k=7):
    t_max = 1e2
    epsilonm = 1e-5
    u = 1e-2
    rho = 2
    u_max = 1e6
    sigma = 1

    v = len(X_list)
    n = np.shape(X_list[0])[1]
    M = np.eye(n) - np.ones((n, n)) / n 

    Z_list = []
    L_list = []
    W_list = []
    H_list = []
    Q_list = []
    E_list = []
    Y1_list = []
    Y2_list = []
    Y3_list = []
    for v_loop in range(v):
        dv = np.shape(X_list[v_loop])[0]
        Z_list.append(np.zeros((n, n)))
        L_list.append(np.zeros((n, n)))
        W_list.append(np.zeros((dv, m)))
        np.random.seed(1)
        H_list.append(np.random.rand(m, n))
        Q_list.append(np.zeros((m, n)))
        E_list.append(np.zeros((dv, n)))
        Y1_list.append(np.zeros((dv, n)))
        Y2_list.append(np.zeros((m, n)))
        Y3_list.append(np.zeros((m, n)))

    t = 0
    loss_list = []
    while t < t_max:
        for v_loop in range(v):
            # Z
            K = np.zeros((n, n))
            for v_loop2 in range(v):
                if v_loop2 != v_loop:
                    K += M @ Z_list[v_loop2].T @ Z_list[v_loop2] @ M
            A_Z = 2 * alpha * np.eye(n) + u * H_list[v_loop].T @ H_list[v_loop]
            B_Z = 2 * lambda2 * K + 2 * lambda3 * L_list[v_loop]
            C_Z = u * H_list[v_loop].T @ (H_list[v_loop] + Y2_list[v_loop] / u)
            Z_list[v_loop] = solve_sylvester(A_Z, B_Z, C_Z)

            # L
            Z_use = abs(Z_list[v_loop])
            Z_use = (Z_use + Z_use.T) / 2
            D = distance.cdist(Z_use.T, Z_use.T)
            S = np.zeros((n, n))
            for j in range(n):
                sort = np.argsort(D[j])
                for i in range(1, k + 1):
                    S[sort[i]][j] = np.exp(-D[sort[i]][j] ** 2 / 2 * sigma ** 2)
            S = (S + S.T) / 2
            L_list[v_loop] = np.diag(S.sum(axis=1)) - S

            # W
            temp_W = X_list[v_loop] - E_list[v_loop] + Y1_list[v_loop] / u
            A_W, N_W, BT_W = np.linalg.svd(H_list[v_loop] @ temp_W.T, full_matrices=False)
            W_list[v_loop] = BT_W.T @ A_W.T

            # H
            A_H = W_list[v_loop].T @ W_list[v_loop] + np.eye(m)
            temp_H = np.eye(n) - Z_list[v_loop]
            B_H = temp_H @ temp_H.T
            C_H = W_list[v_loop].T @ (X_list[v_loop] - E_list[v_loop] + Y1_list[v_loop] / u) - \
                  1 / u * Y2_list[v_loop] @ temp_H.T + Q_list[v_loop] - Y3_list[v_loop] / u
            H_list[v_loop] = solve_sylvester(A_H, B_H, C_H)

            # Q
            A_Q, N_Q, BT_Q = np.linalg.svd(H_list[v_loop] + Y3_list[v_loop] / u, full_matrices=False)
            N = np.maximum(N_Q - 1 / u, 0)
            Q_list[v_loop] = A_Q @ np.diag(N) @ BT_Q

            # E
            B_E = X_list[v_loop] - W_list[v_loop] @ H_list[v_loop] + Y1_list[v_loop] / u
            for j in range(n):
                temp_E = np.linalg.norm(B_E[:, j])
                if temp_E > lambda1 / u:
                    E_list[v_loop][:, j] = (1 - lambda1 / u / temp_E) * B_E[:, j]
                else:
                    E_list[v_loop][:, j] = 0

            Y1_list[v_loop] += u * (X_list[v_loop] - W_list[v_loop] @ H_list[v_loop] - E_list[v_loop])
            Y2_list[v_loop] += u * (H_list[v_loop] - H_list[v_loop] @ Z_list[v_loop])
            Y3_list[v_loop] += u * (H_list[v_loop] - Q_list[v_loop])

        u = np.minimum(u_max, u * rho)

        loss = np.zeros(v)
        for v_loop in range(v):
            temp1 = np.linalg.norm(X_list[v_loop] - W_list[v_loop] @ H_list[v_loop] - E_list[v_loop], ord=np.inf)
            temp2 = np.linalg.norm(H_list[v_loop] - H_list[v_loop] @ Z_list[v_loop], ord=np.inf)
            temp3 = np.linalg.norm(H_list[v_loop] - Q_list[v_loop], ord=np.inf)
            temp_max = np.maximum(temp1, temp2)
            loss[v_loop] = np.maximum(temp3, temp_max)
        loss = np.max(loss)
        loss_list.append(loss)
        t += 1
        # print("t:{} loss:{}".format(t, loss))
        if loss < epsilonm:
            break

    return Z_list


