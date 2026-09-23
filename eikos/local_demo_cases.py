"""Local demo cases (shared by local_demo.py/PyTorch and mlx_decide.py/MLX)."""

CASES = [
    ("EN trading", "Order ticket #A-2231. Client: retail, risk profile moderate. Request: BUY 1,500 shares of XYZ at market. "
     "Account equity USD 48,000. Last price USD 41.20. Firm rule 4.2: for moderate-risk retail clients a single order may not "
     "exceed 50% of account equity unless a supervisor approves it in writing. Supervisor approvals on file: none.",
     {"type": "noul", "instructions": "Under rule 4.2, can this order be executed as submitted?",
      "criteria": {"true": "complies with rule 4.2", "false": "breaches rule 4.2"}}),
    ("PT refund", "Pedido de reembolso #88213. Compra em 02/09/2026, R$ 349,90. Política: reembolso integral em até 7 dias "
     "corridos da ENTREGA se o produto estiver lacrado. Entrega registrada em 05/09/2026. Pedido aberto em 11/09/2026. "
     "Foto mostra a embalagem lacrada.",
     {"type": "noul", "instructions": "O cliente tem direito ao reembolso integral?",
      "criteria": {"true": "a política garante o reembolso", "false": "a política não garante o reembolso"}}),
    ("ES payments (language never trained)", "Transferencia #TX-5510 por EUR 18.400 a un proveedor nuevo en Panamá. Política: "
     "toda transferencia internacional superior a EUR 15.000 a un beneficiario nuevo requiere verificación por llamada "
     "(callback) antes de liberar. Registro: callback no realizado.",
     {"type": "choice", "instructions": "¿Qué debe hacer el equipo de pagos?",
      "criteria": {"release": "liberar la transferencia", "hold_for_callback": "retener hasta hacer el callback",
                   "reject": "rechazar definitivamente"}}),
]

MULTI = [CASES[0][2],
      {"type": "choice", "instructions": "What should the desk do?",
       "criteria": {"execute": "send as submitted", "request_approval": "hold and ask a supervisor",
                    "reduce_size": "cut the order to the allowed size", "reject": "refuse the order"}},
      {"type": "score", "instructions": "How risky is this ticket for the firm?",
       "criteria": ["low", "moderate", "high", "critical"]}]
