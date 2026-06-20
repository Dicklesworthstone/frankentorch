# frankentorch-o888p Provenance / Legal Note

This change is original implementation work in `ft-kernel-cpu`.

No third-party source code was copied. The implementation uses existing local FrankenTorch max-pool code structure and the project benchmark harness. The external comparison target is PyTorch 2.12 CPU behavior/performance, used only as a runtime oracle through the existing local benchmark script.

Patent/IP risk for this lever is low: the committed technique is a common loop-shape specialization and linear sidecar traversal, not a novel patented algorithm.
