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
                {"name": "layout"},
                {"name": "periodicity"},
                {"name": "lattice"},
            ],
            "parameterTree": {
                "next": {
                    "Ising": {
                        "next": {
                            "1x4": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "transverse-field-ising-model-ising-1x4-chain-open"
                                            },
                                            "closed": {
                                                "value": "transverse-field-ising-model-ising-1x4-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "1x8": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "transverse-field-ising-model-ising-1x8-chain-open"
                                            },
                                            "closed": {
                                                "value": "transverse-field-ising-model-ising-1x8-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x2": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "transverse-field-ising-model-ising-2x2-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "transverse-field-ising-model-ising-2x2-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x4": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "transverse-field-ising-model-ising-2x4-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "transverse-field-ising-model-ising-2x4-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x8": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "transverse-field-ising-model-ising-2x8-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "transverse-field-ising-model-ising-2x8-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "4x4": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "transverse-field-ising-model-ising-4x4-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "transverse-field-ising-model-ising-4x4-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "1x16": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "transverse-field-ising-model-ising-1x16-chain-open"
                                            },
                                            "closed": {
                                                "value": "transverse-field-ising-model-ising-1x16-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "open": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
                                },
                                "default": "chain",
                            },
                        },
                        "default": "open",
                    },
                    "Heisenberg": {
                        "next": {
                            "1x4": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "xxz-heisenberg-model-heisenberg-1x4-chain-open"
                                            },
                                            "closed": {
                                                "value": "xxz-heisenberg-model-heisenberg-1x4-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "1x8": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "xxz-heisenberg-model-heisenberg-1x8-chain-open"
                                            },
                                            "closed": {
                                                "value": "xxz-heisenberg-model-heisenberg-1x8-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x2": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "xxz-heisenberg-model-heisenberg-2x2-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "xxz-heisenberg-model-heisenberg-2x2-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x4": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "xxz-heisenberg-model-heisenberg-2x4-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "xxz-heisenberg-model-heisenberg-2x4-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x8": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "xxz-heisenberg-model-heisenberg-2x8-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "xxz-heisenberg-model-heisenberg-2x8-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "4x4": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "xxz-heisenberg-model-heisenberg-4x4-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "xxz-heisenberg-model-heisenberg-4x4-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "1x16": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "xxz-heisenberg-model-heisenberg-1x16-chain-open"
                                            },
                                            "closed": {
                                                "value": "xxz-heisenberg-model-heisenberg-1x16-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "open": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
                                },
                                "default": "chain",
                            },
                        },
                        "default": "open",
                    },
                    "BoseHubbard": {
                        "next": {
                            "1x4": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "bose-hubbard-model-bosehubbard-1x4-chain-open"
                                            },
                                            "closed": {
                                                "value": "bose-hubbard-model-bosehubbard-1x4-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "1x8": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "bose-hubbard-model-bosehubbard-1x8-chain-open"
                                            },
                                            "closed": {
                                                "value": "bose-hubbard-model-bosehubbard-1x8-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x2": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "bose-hubbard-model-bosehubbard-2x2-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "bose-hubbard-model-bosehubbard-2x2-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x4": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "bose-hubbard-model-bosehubbard-2x4-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "bose-hubbard-model-bosehubbard-2x4-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "open": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
                                },
                                "default": "chain",
                            },
                        },
                        "default": "open",
                    },
                    "FermiHubbard": {
                        "next": {
                            "1x4": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "fermi-hubbard-model-fermihubbard-1x4-chain-open"
                                            },
                                            "closed": {
                                                "value": "fermi-hubbard-model-fermihubbard-1x4-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "1x8": {
                                "next": {
                                    "chain": {
                                        "next": {
                                            "open": {
                                                "value": "fermi-hubbard-model-fermihubbard-1x8-chain-open"
                                            },
                                            "closed": {
                                                "value": "fermi-hubbard-model-fermihubbard-1x8-chain-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x2": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "fermi-hubbard-model-fermihubbard-2x2-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "fermi-hubbard-model-fermihubbard-2x2-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "2x4": {
                                "next": {
                                    "rectangular": {
                                        "next": {
                                            "open": {
                                                "value": "fermi-hubbard-model-fermihubbard-2x4-rectangular-open"
                                            },
                                            "closed": {
                                                "value": "fermi-hubbard-model-fermihubbard-2x4-rectangular-closed"
                                            },
                                        },
                                        "default": None,
                                    }
                                },
                                "default": None,
                            },
                            "open": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
                                },
                                "default": "chain",
                            },
                            "closed": {
                                "next": {
                                    "chain": {"next": {}, "default": "1x4"},
                                    "rectangular": {"next": {}, "default": "2x2"},
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
