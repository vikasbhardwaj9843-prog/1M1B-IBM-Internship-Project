# UN SDG Alignment

NeuralZip is built with an explicit sustainability mandate. This document maps the
project's technical features to the specific targets of the two UN Sustainable Development
Goals it supports.

---

## UN SDG 9 — Industry, Innovation and Infrastructure

> *"Build resilient infrastructure, promote inclusive and sustainable industrialisation,
> and foster innovation."*

### Target 9.4
> *"By 2030, upgrade infrastructure and retrofit industries to make them sustainable,
> with increased resource-use efficiency and greater adoption of clean and
> environmentally sound technologies and industrial processes."*

**NeuralZip contribution:**

| Feature | How it addresses 9.4 |
|---|---|
| Neural compression reduces storage footprint by 3–5× | Directly reduces demand for new storage hardware — HDDs, SSDs, and tape — thereby reducing manufacturing energy and materials |
| Near-lossless mode (PSNR > 40 dB) | Enables institutions to retire raw-pixel archives without sacrificing data quality |
| Lossless mode with residual correction | Ensures regulatory compliance (medical records, legal archives) — enabling adoption in high-stakes infrastructure |
| Open-source, config-driven design | Zero-barrier deployment; any institution globally can integrate it into existing archival pipelines without vendor dependency |

### Target 9.b
> *"Support domestic technology development, research and innovation in developing countries,
> including ensuring a conducive policy and regulatory environment for, inter alia,
> industrial diversification and value addition to commodities."*

**NeuralZip contribution:**

| Feature | How it addresses 9.b |
|---|---|
| MIT-licensed, fully open-source | Available to research institutions and governments in any country |
| No proprietary dependencies | Runs on commodity hardware; no cloud vendor lock-in |
| Config-driven, documented architecture | Enables local adaptation and extension without specialist AI knowledge |
| PyTorch-based | Leverages the world's most widely taught deep learning framework — low barrier to local development |

---

## UN SDG 12 — Responsible Consumption and Production

> *"Ensure sustainable consumption and production patterns."*

### Target 12.2
> *"By 2030, achieve sustainable management and efficient use of natural resources."*

**NeuralZip contribution:**

Digital storage infrastructure consumes natural resources at every level:
electricity generation, rare-earth materials in storage chips, water for data centre
cooling. Reducing storage footprint directly reduces all of these downstream consumptions.

| Metric | Conventional Archival | NeuralZip Near-lossless |
|---|---|---|
| Storage per 1M grayscale 1024² images | ~1 TB | ~200–300 GB (3–5× reduction) |
| Equivalent HDD count (4 TB drives) | 1 drive per 4M images | ~1 drive per 16–20M images |
| Data centre cooling overhead | Proportional to storage | Reduced proportionally |

### Target 12.5
> *"By 2030, substantially reduce waste generation through prevention, reduction,
> reuse, and recycling."*

**NeuralZip contribution:**

Smaller archival footprints directly reduce the rate at which storage hardware is
purchased, powered, and ultimately discarded as e-waste:

| Feature | Waste reduction pathway |
|---|---|
| 3–5× storage reduction | Fewer HDDs/SSDs manufactured per petabyte of archive |
| Longer hardware refresh cycles | Less frequent hardware replacement → less e-waste |
| Lossless mode preserves original data | No need to re-acquire or re-scan archived data |
| Float16 latent option (~0.5 MB/image) | Further halves storage; reduces energy for I/O operations |

---

## Summary Alignment Table

| SDG Target | Description | NeuralZip Feature |
|---|---|---|
| 9.4 | Sustainable infrastructure upgrade | Neural compression reduces storage hardware demand |
| 9.4 | Resource-use efficiency | 3–5× reduction in storage footprint |
| 9.b | Domestic tech development | Open-source, no proprietary dependencies |
| 9.b | Accessible innovation | Config-driven, PyTorch-based, well-documented |
| 12.2 | Efficient use of natural resources | Fewer drives, less electricity, less cooling water |
| 12.5 | Reduce waste generation | Fewer drives manufactured and discarded |

---

## References

- United Nations SDG 9: https://sdgs.un.org/goals/goal9
- United Nations SDG 12: https://sdgs.un.org/goals/goal12
- IEA Data Centres and Data Transmission Networks: https://www.iea.org/reports/data-centres-and-data-transmission-networks
