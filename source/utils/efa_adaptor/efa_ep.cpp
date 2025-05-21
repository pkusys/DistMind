#include "efa_ep.h"

namespace trans {

EFAEndpoint::EFAEndpoint(std::string nickname) { 
  this->nickname = nickname; 
  init_res();
};

int EFAEndpoint::init_res() {
  struct fi_info *hints;
  struct fi_cq_attr txcq_attr, rxcq_attr;
  struct fi_av_attr av_attr;
  int err;
  std::string provider = "verbs;ofi_rxm";
  std::string device = "mlx5_0";

  hints = fi_allocinfo();
  if (!hints)
    std::cerr << "fi_allocinfo err " << -ENOMEM << "\n";

  // clear all buffers
  memset(&txcq_attr, 0, sizeof(txcq_attr));
  memset(&rxcq_attr, 0, sizeof(rxcq_attr));
  memset(&av_attr, 0, sizeof(av_attr));

  // get provider
  hints->ep_attr->type = FI_EP_RDM;
  hints->fabric_attr->prov_name = strdup(provider.c_str());
  hints->caps = FI_MSG | FI_TAGGED;
  // SAS
  hints->rx_attr->msg_order = FI_ORDER_SAS;
  hints->tx_attr->msg_order = FI_ORDER_SAS;
  // device
  hints->domain_attr->name = strdup(device.c_str());
  hints->domain_attr->av_type = FI_AV_TABLE;
  hints->domain_attr->mr_mode = FI_MR_BASIC;
  // std::cout << "===========================================Before check" << std::endl;
  err = fi_getinfo(FI_VERSION(1, 9), NULL, NULL, 0, hints, &fi);
  // std::cout << "===========================================After check" << std::endl;
  if (err < 0)
    std::cerr << "fi_getinfo err " << err << "\n";

  // fi_freeinfo(hints);
  std::cout << "Using OFI device: " << fi->fabric_attr->name << "\n";
  std::cout << "Using OFI provider: " << fi->fabric_attr->prov_name << "\n";
  std::cout << "Using MR Basic: " << (fi->domain_attr->mr_mode & FI_MR_BASIC) << "\n";

  // init fabric, domain, address-vector,
  err = fi_fabric(fi->fabric_attr, &fabric, NULL);
  if (err < 0)
    std::cerr << "fi_fabric err " << err << "\n";
  err = fi_domain(fabric, fi, &domain, NULL);
  if (err < 0)
    std::cerr << "fi_domain err " << err << "\n";

  av_attr.type = fi->domain_attr->av_type;
  av_attr.count = 1;
  err = fi_av_open(domain, &av_attr, &av, NULL);
  if (err < 0)
    std::cerr << "fi_av_open err " << err << "\n";

  // open complete queue
  txcq_attr.format = FI_CQ_FORMAT_TAGGED;
  txcq_attr.size = fi->tx_attr->size;
  rxcq_attr.format = FI_CQ_FORMAT_TAGGED;
  rxcq_attr.size = fi->rx_attr->size;
  err = fi_cq_open(domain, &txcq_attr, &txcq, NULL);
  if (err < 0)
    std::cerr << "fi_txcq_open err " << err << "\n";
  err = fi_cq_open(domain, &rxcq_attr, &rxcq, NULL);
  if (err < 0)
    std::cerr << "fi_rxcq_open err " << err << "\n";
  std::cout << "--- fi->tx_attr-size: " 
            << fi->tx_attr->size << "\n"
            << "--- fi->rx_attr->size: "
            << fi->rx_attr->size << "\n";

  // open endpoint
  err = fi_endpoint(domain, fi, &ep, NULL);
  if (err < 0)
    std::cerr << "fi_endpoint err " << err << "\n";

  // bind complete queue, address vector to endpoint
  err = fi_ep_bind(ep, (fid_t)txcq, FI_SEND | FI_TRANSMIT);
  if (err < 0)
    std::cerr << "fi_ep_bind txcq err " << err << "\n";
  err = fi_ep_bind(ep, (fid_t)rxcq, FI_RECV);
  printf("%s rxcq : %p\n", this->nickname.c_str(), rxcq);
  if (err < 0)
    std::cerr << "fi_ep_bind rxcq err " << err << "\n";
  // printf("%s bind txcq %p; rxcq %p;\n", nickname.c_str(), txcq, rxcq);
  err = fi_ep_bind(ep, (fid_t)av, 0);
  if (err < 0)
    std::cerr << "fi_ep_bind av err " << err << "\n";

  // enable endpoint
  err = fi_enable(ep);
  if (err < 0)
    std::cerr << "fi_enable err " << err << "\n";
  ep_ready = true;
  return ep_ready;
};

void EFAEndpoint::get_name(char *name_buf, int size) {
  int err = 0;
  size_t len = size;
  err = fi_getname((fid_t)ep, name_buf, &len);
  if (err < 0)
    std::cerr << "fi_getname err " << err << "\n";
};

void EFAEndpoint::insert_peer_address(char *addr) {
  int ret = 0;
  ret = fi_av_insert(av, addr, 1, &peer_addr, 0, NULL);
  if (ret != 1)
    std::cerr << "fi_av_insert " << ret << "\n";
};

EFAEndpoint::~EFAEndpoint() {
  fi_close((fid_t)ep);
  fi_close((fid_t)txcq);
  fi_close((fid_t)rxcq);
  fi_close((fid_t)av);
  fi_close((fid_t)domain);
  fi_close((fid_t)fabric);
  fi_freeinfo(fi);
};

}; // namespace trans