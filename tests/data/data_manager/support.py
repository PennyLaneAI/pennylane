"""Test support for mocking GraphQL queries"""

_list_attrs_resp = {
    "data": {
        "datasetClass": {
            "attributes": [
                {"name": "basis_rot_groupings"},
                {"name": "basis_rot_samples"},
                {"name": "dipole_op"},
                {"name": "fci_energy"},
                {"name": "fci_spectrum"},
                {"name": "hamiltonian"},
                {"name": "hf_state"},
                {"name": "molecule"},
                {"name": "number_op"},
                {"name": "optimal_sector"},
                {"name": "paulix_ops"},
                {"name": "qwc_groupings"},
                {"name": "qwc_samples"},
                {"name": "sparse_hamiltonian"},
                {"name": "spin2_op"},
                {"name": "spinz_op"},
                {"name": "symmetries"},
                {"name": "tapered_dipole_op"},
                {"name": "tapered_hamiltonian"},
                {"name": "tapered_hf_state"},
                {"name": "tapered_num_op"},
                {"name": "tapered_spin2_op"},
                {"name": "tapered_spinz_op"},
                {"name": "vqe_energy"},
                {"name": "vqe_gates"},
                {"name": "vqe_params"},
            ]
        }
    }
}

_get_urls_resp = {
    "data": {
        "datasetClass": {
            "datasets": [
                {
                    "id": "h2_sto-3g_0.46",
                    "downloadUrl": "https://cloud.pennylane.ai/datasets/download/h2_sto-3g_0.46",
                },
                {
                    "id": "h2_sto-3g_1.16",
                    "downloadUrl": "https://cloud.pennylane.ai/datasets/download/h2_sto-3g_1.16",
                },
                {
                    "id": "h2_sto-3g_1.0",
                    "downloadUrl": "https://cloud.pennylane.ai/datasets/download/h2_sto-3g_1.0",
                },
            ]
        }
    }
}

_dataclass_ids = {"data": {"datasetClasses": [{"id": "other"}, {"id": "qchem"}, {"id": "qspin"}]}}

_error_response = {"data": None, "errors": [{"message": "Mock error message."}]}

_parameter_tree = {
    "data": {
        "datasetClass": {
            "attributes": [
                {"name": "ground_energies"},
                {"name": "ground_states"},
                {"name": "hamiltonians"},
                {"name": "num_phases"},
                {"name": "order_params"},
                {"name": "parameters"},
                {"name": "shadow_basis"},
                {"name": "shadow_meas"},
                {"name": "spin_system"},
            ],
            "parameters": [
                {"name": "sysname"},
                {"name": "periodicity"},
                {"name": "lattice"},
                {"name": "layout"},
            ],
            "parameterTree": {
                "next": {
                    "Ising": {
                        "next": {
                            "open": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "transverse-field-ising-model-ising-open-chain-1x4",
                                            "1x8": "transverse-field-ising-model-ising-open-chain-1x8",
                                            "1x16": "transverse-field-ising-model-ising-open-chain-1x16",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "transverse-field-ising-model-ising-open-rectangular-2x2",
                                            "2x4": "transverse-field-ising-model-ising-open-rectangular-2x4",
                                            "2x8": "transverse-field-ising-model-ising-open-rectangular-2x8",
                                            "4x4": "transverse-field-ising-model-ising-open-rectangular-4x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "transverse-field-ising-model-ising-closed-chain-1x4",
                                            "1x8": "transverse-field-ising-model-ising-closed-chain-1x8",
                                            "1x16": "transverse-field-ising-model-ising-closed-chain-1x16",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "transverse-field-ising-model-ising-closed-rectangular-2x2",
                                            "2x4": "transverse-field-ising-model-ising-closed-rectangular-2x4",
                                            "2x8": "transverse-field-ising-model-ising-closed-rectangular-2x8",
                                            "4x4": "transverse-field-ising-model-ising-closed-rectangular-4x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                        },
                        "default": "open",
                    },
                    "Heisenberg": {
                        "next": {
                            "open": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "xxz-heisenberg-model-heisenberg-open-chain-1x4",
                                            "1x8": "xxz-heisenberg-model-heisenberg-open-chain-1x8",
                                            "1x16": "xxz-heisenberg-model-heisenberg-open-chain-1x16",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "xxz-heisenberg-model-heisenberg-open-rectangular-2x2",
                                            "2x4": "xxz-heisenberg-model-heisenberg-open-rectangular-2x4",
                                            "2x8": "xxz-heisenberg-model-heisenberg-open-rectangular-2x8",
                                            "4x4": "xxz-heisenberg-model-heisenberg-open-rectangular-4x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "xxz-heisenberg-model-heisenberg-closed-chain-1x4",
                                            "1x8": "xxz-heisenberg-model-heisenberg-closed-chain-1x8",
                                            "1x16": "xxz-heisenberg-model-heisenberg-closed-chain-1x16",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "xxz-heisenberg-model-heisenberg-closed-rectangular-2x2",
                                            "2x4": "xxz-heisenberg-model-heisenberg-closed-rectangular-2x4",
                                            "2x8": "xxz-heisenberg-model-heisenberg-closed-rectangular-2x8",
                                            "4x4": "xxz-heisenberg-model-heisenberg-closed-rectangular-4x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                        },
                        "default": "open",
                    },
                    "BoseHubbard": {
                        "next": {
                            "open": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "bose-hubbard-model-bosehubbard-open-chain-1x4",
                                            "1x8": "bose-hubbard-model-bosehubbard-open-chain-1x8",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "bose-hubbard-model-bosehubbard-open-rectangular-2x2",
                                            "2x4": "bose-hubbard-model-bosehubbard-open-rectangular-2x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "bose-hubbard-model-bosehubbard-closed-chain-1x4",
                                            "1x8": "bose-hubbard-model-bosehubbard-closed-chain-1x8",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "bose-hubbard-model-bosehubbard-closed-rectangular-2x2",
                                            "2x4": "bose-hubbard-model-bosehubbard-closed-rectangular-2x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                        },
                        "default": "open",
                    },
                    "FermiHubbard": {
                        "next": {
                            "open": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "fermi-hubbard-model-fermihubbard-open-chain-1x4",
                                            "1x8": "fermi-hubbard-model-fermihubbard-open-chain-1x8",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "fermi-hubbard-model-fermihubbard-open-rectangular-2x2",
                                            "2x4": "fermi-hubbard-model-fermihubbard-open-rectangular-2x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "1x4": "fermi-hubbard-model-fermihubbard-closed-chain-1x4",
                                            "1x8": "fermi-hubbard-model-fermihubbard-closed-chain-1x8",
                                        },
                                        "default": "1x4",
                                    },
                                    "rectangular": {
                                        "next": {
                                            "2x2": "fermi-hubbard-model-fermihubbard-closed-rectangular-2x2",
                                            "2x4": "fermi-hubbard-model-fermihubbard-closed-rectangular-2x4",
                                        },
                                        "default": "2x2",
                                    },
                                },
                                "default": "chain",
                            },
                        },
                        "default": "open",
                    },
                },
                "default": None,
            },
        }
    }
}
