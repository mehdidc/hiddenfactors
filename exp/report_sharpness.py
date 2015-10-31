import matplotlib as mpl
import numpy as np
mpl.use('Agg')
import matplotlib.pyplot as plt

from lightexperiments.light import Light
light =   Light()
light.launch()
reports = list(light.db.find({"custom_id": "06326877-5382-4ed9-b017-d7a611b51039", "tags": ["decorrelation"]}))
reports = sorted(reports, key=lambda r:r["latent_size"])
latent = [r["latent_size"] for r in reports]
print(latent)
print(len(reports))

sharpness_ratios = [(np.array(r["sharpness_test_rec"]) / np.array(r["sharpness_test"])) for r in reports]
sharpness_ratios_mean = [np.mean(a) for a in sharpness_ratios]

print((sharpness_ratios[0]>1).sum())

sharpness_ratios_std = [np.std(a) for a in sharpness_ratios]

plt.errorbar(latent, sharpness_ratios_mean, yerr=sharpness_ratios_std)
plt.xlabel("latent size")
plt.ylabel("sharpness ratio mean")
plt.title("latent size / sharpness ratios")
plt.savefig("fig1.png")
plt.show()

plt.clf()
plt.plot(latent, [r["rec_valid"][-1] for r in reports])
plt.xlabel("latent size")
plt.ylabel("test rec")
plt.title("latent size / rec test error")
plt.savefig("fig2.png")
plt.show()

plt.clf()
gen_mean = [ np.mean([l for l in r["sharpness_generated_per_latent"]]) for r in reports]
gen_std = [ np.std([l for l in r["sharpness_generated_per_latent"]]) for r in reports]

plt.errorbar(latent, gen_mean, yerr=gen_std)
plt.xlabel("latent size")
plt.ylabel("mean generated sharpness")
plt.title("latent size / mean generated sharpness")
plt.savefig("fig3.png")
light.close()
