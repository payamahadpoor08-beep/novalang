#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int w = 96, h = 96, gens = 120;
    int n = w * h;
    int *cur = calloc(n, sizeof(int));
    int *nxt = calloc(n, sizeof(int));
    long long seed = 1234567;
    for (int i = 0; i < n; i++) {
        seed = (seed * 1103515245 + 12345) & 2147483647;
        cur[i] = (seed >> 16) & 1;
    }
    for (int g = 0; g < gens; g++) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int c = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx != 0 || dy != 0) {
                            int nx = (x + dx + w) % w;
                            int ny = (y + dy + h) % h;
                            c += cur[ny * w + nx];
                        }
                    }
                }
                int idx = y * w + x;
                int alive = cur[idx];
                int cell = 0;
                if (alive == 1) {
                    if (c == 2 || c == 3) cell = 1;
                } else {
                    if (c == 3) cell = 1;
                }
                nxt[idx] = cell;
            }
        }
        for (int i = 0; i < n; i++) cur[i] = nxt[i];
    }
    long long pop = 0;
    for (int i = 0; i < n; i++) pop += cur[i];
    printf("%lld\n", pop);
    return 0;
}
