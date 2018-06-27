#ifndef PROTOCOLPARTY_H_
#define PROTOCOLPARTY_H_

#include <stdlib.h>

#include <libscapi/include/primitives/Matrix.hpp>
#include <libscapi/include/cryptoInfra/Protocol.hpp>
#include <libscapi/include/circuits/ArithmeticCircuit.hpp>
#include <libscapi/include/infra/Measurement.hpp>
#include <vector>
#include <bitset>
#include <iostream>
#include <fstream>
#include <chrono>
#include <libscapi/include/primitives/Mersenne.hpp>
#include "ProtocolTimer.h"
#include <libscapi/include/comm/MPCCommunication.hpp>
#include <libscapi/include/infra/Common.hpp>
#include <libscapi/include/primitives/Prg.hpp>
#include "HashEncrypt.h"
#include <emmintrin.h>
#include <thread>

#include "comm_client_cb_api.h"
#include "comm_client_factory.h"
#include "comm_client.h"

#include <log4cpp/Category.hh>
#include <log4cpp/FileAppender.hh>
#include <log4cpp/SimpleLayout.hh>
#include <log4cpp/RollingFileAppender.hh>
#include <log4cpp/SimpleLayout.hh>
#include <log4cpp/BasicLayout.hh>
#include <log4cpp/PatternLayout.hh>

void init_log(const char * a_log_file, const char * a_log_dir,
		const int log_level, const char * logcat);

#define flag_print false
#define flag_print_timings true
#define flag_print_output true

using namespace std;
using namespace std::chrono;

template<class FieldType>
class ProtocolParty: public Protocol,
		public HonestMajority,
		MultiParty,
		public comm_client_cb_api {

private:

	/**
	 * N - number of parties
	 * M - number of gates
	 * T - number of malicious
	 */

	int N, M, T, m_partyId;
	int times; //number of times to run the run function
	int iteration; //number of the current iteration

	Measurement* timer;
	VDM<FieldType> matrix_vand;
	TemplateField<FieldType> *field;
	vector<FieldType> randomTAnd2TShares;
	vector<FieldType> randomSharesArray;
	vector<FieldType> bigR;
	vector<byte> h; //a string accumulated that should be hashed in the comparing views function.

	ProtocolTimer* protocolTimer;
	int currentCirciutLayer = 0;
	int offset = 0;
	int randomSharesOffset = 0;

	string s;
	int numOfInputGates, numOfOutputGates;
	string inputsFile, outputFile;
	vector<FieldType> beta;
	HIM<FieldType> matrix_for_interpolate;
	HIM<FieldType> matrix_for_t;
	HIM<FieldType> matrix_for_2t;
	vector<FieldType> y_for_interpolate;

	HIM<FieldType> matrix_him;

	VDMTranspose<FieldType> matrix_vand_transpose;

	HIM<FieldType> m;

	boost::asio::io_service io_service;
	ArithmeticCircuit circuit;
	vector<FieldType> gateValueArr; // the value of the gate (for my input and output gates)
	vector<FieldType> gateShareArr; // my share of the gate (for all gates)
	vector<FieldType> alpha; // N distinct non-zero field elements

	vector<long> myInputs;

	//****************************************************************************************************//
	std::string logcat, logcat_tim, logcat_out;
	comm_client * m_cc;

	typedef struct __peer_t {
		size_t id;
		bool conn;
		std::vector<u_int8_t> data;

		__peer_t() :
				id(0), conn(false) {
		}
	} peer_t;
	std::vector<peer_t> m_parties;

	typedef enum {
		comm_evt_nil = 0, comm_evt_conn, comm_evt_msg
	} comm_evt_type_t;

	class comm_evt {
	public:
		comm_evt() :
				type(comm_evt_nil), party_id(-1) {
		}
		virtual ~comm_evt() {
		}
		comm_evt_type_t type;
		unsigned int party_id;
	};

	class comm_conn_evt: public comm_evt {
	public:
		bool connected;
	};

	class comm_msg_evt: public comm_evt {
	public:
		std::vector<u_int8_t> msg;
	};

	pthread_mutex_t m_qlock, m_elock;
	pthread_cond_t m_comm_e;
	std::list<comm_evt *> m_comm_q;

	void push_comm_event(comm_evt * evt);
	void report_party_comm(const size_t party_id, const bool comm);
	void process_network_events();
	void wait_for_peer_connections();
	//****************************************************************************************************//

public:
//    ProtocolParty(int n, int id,string fieldType, string inputsFile, string outputFile, string circuitFile,
//             int groupID = 0);
	ProtocolParty(int argc, char* argv[]);

	void roundFunctionSync(vector<vector<byte>> &sendBufs,
			vector<vector<byte>> &recBufs, int round);
	void roundFunctionSyncForP1(vector<byte> &myShare,
			vector<vector<byte>> &recBufs);

	//comm_client_cb_api
	void on_comm_up_with_party(const unsigned int party_id);
	void on_comm_down_with_party(const unsigned int party_id);
	void on_comm_message(const unsigned int src_id, const unsigned char * msg,
			const size_t size);

	int counter = 0;

	/**
	 * This method runs the protocol:
	 * 1. Preparation Phase
	 * 2. Input Phase
	 * 3. Computation Phase
	 * 4. Verification Phase
	 * 5. Output Phase
	 */
	void run() override;

	bool hasOffline() {
		return true;
	}

	bool hasOnline() override {
		return true;
	}

	/**
	 * This method runs the protocol:
	 * Preparation Phase
	 */
	void runOffline() override;

	/**
	 * This method runs the protocol:
	 * Input Phase
	 * Computation Phase
	 * Verification Phase
	 * Output Phase
	 */
	void runOnline() override;

	/**
	 * This method reads text file and inits a vector of Inputs according to the file.
	 */
	void readMyInputs();

	/**
	 * We describe the protocol initialization.
	 * In particular, some global variables are declared and initialized.
	 */
	void initializationPhase();

	/**
	 * A random double-sharing is a pair of two sharings of the same random value, where the one sharing is
	 * of degree t, and the other sharing is of degree 2t. Such random double-sharing are of big help in the
	 * multiplication protocol.
	 * We use hyper-invertible matrices to generate random double-sharings. The basic idea is as follows:
	 * Every party generates one random double-sharing. These n double-sharings are processes through a
	 * hyper-invertible matrix. From the resulting n double-sharings, t are checked to be valid (correct degree,
	 * same secret), and t are then kept as “good” double-sharings. This is secure due to the diversion property
	 * of hyper-invertible matrix: We know that n − t of the input double-sharings are good. So, if there are t
	 * valid output double-sharings, then all double-sharings must be valid. Furthermore, the adversary knows
	 * his own up to t input double-sharings, and learns t output double sharings. So, n − 2t output double
	 * sharings are random and unknown to the adversary.
	 * For the sake of efficiency, we do not publicly reconstruct t of the output double-sharings. Rather, we
	 * reconstruct 2t output double sharings, each to one dedicated party only. At least t of these parties are
	 * honest and correctly validate the reconstructed double-sharing.
	 *
	 * The goal of this phase is to generate “enough” double-sharings to evaluate the circuit. The double-
	 * sharings are stored in a buffer SharingBuf , where alternating a degree-t and a degree-2t sharing (of the same secret)
	 * is stored (more precisely, a share of each such corresponding sharings is stored).
	 * The creation of double-sharings is:
	 *
	 * Protocol Generate-Double-Sharings:
	 * 1. ∀i: Pi selects random value x-(i) and computes degree-t shares x1-(i) and degree-2t shares x2-(i).
	 * 2. ∀i,j: Pi sends the shares x1,j and X2,j to party Pj.
	 * 3. ∀j: Pj applies a hyper-invertible matrix M on the received shares, i.e:
	 *      (y1,j,..., y1,j) = M(x1,j,...,x1,j)
	 *      (y2,j,...,y2,j) = M (x2,j,...,x2,)
	 * 4. ∀j, ∀k ≤ 2t: Pj sends y1,j and y2,j to Pk.
	 * 5. ∀k ≤ 2t: Pk checks:
	 *      • that the received shares (y1,1,...,y1,n) are t-consistent,
	 *      • that the received shares (y2,1,...,y2,n) are 2t-consistent, and
	 *      • that both sharings interpolate to the same secret.
	 *
	 * We use this algorithm, but extend it to capture an arbitrary number of double-sharings.
	 * This is, as usual, achieved by processing multiple buckets in parallel.
	 */
	bool preparationPhase();

	/**
	 * This protocol is secure only in the presence of a semi-honest adversary.
	 */
	void DNHonestMultiplication(FieldType *a, FieldType *b,
			vector<FieldType> &cToFill, int numOfTrupples);

	void offlineDNForMultiplication(int numOfTriples);

	/**
	 * The input phase proceeds in two steps:
	 * First, for each input gate, the party owning the input creates shares for that input by choosing a random coefficients for the polynomial
	 * Then, all the shares are sent to the relevant party
	 */
	void inputPhase();
	void inputVerification(vector<FieldType> &inputShares);

	void generateRandomShares(int numOfRandoms,
			vector<FieldType> &randomElementsToFill);
	void getRandomShares(int numOfRandoms,
			vector<FieldType> &randomElementsToFill);
	void generateRandomSharesWithCheck(int numOfRnadoms,
			vector<FieldType>& randomElementsToFill);
	void generateRandom2TAndTShares(int numOfRandomPairs,
			vector<FieldType>& randomElementsToFill);

	/**
	 * Check whether given points lie on polynomial of degree d.
	 * This check is performed by interpolating x on the first d + 1 positions of α and check the remaining positions.
	 */
	bool checkConsistency(vector<FieldType>& x, int d);

	FieldType reconstructShare(vector<FieldType>& x, int d);

	void openShare(int numOfRandomShares, vector<FieldType> &Shares,
			vector<FieldType> &secrets);

	/**
	 * Process all multiplications which are ready.
	 * Return number of processed gates.
	 */
	int processMultiplications(int lastMultGate);

	int processMultDN(int indexInRandomArray);

	int processNotMult();

	/**
	 * Walk through the circuit and evaluate the gates. Always take as many gates at once as possible,
	 * i.e., all gates whose inputs are ready.
	 * We first process all random gates, then alternately process addition and multiplication gates.
	 */
	void computationPhase(HIM<FieldType> &m);

	/**
	 * The cheap way: Create a HIM from the αi’s onto ZERO (this is actually a row vector), and multiply
	 * this HIM with the given x-vector (this is actually a scalar product).
	 * The first (and only) element of the output vector is the secret.
	 */
	FieldType interpolate(vector<FieldType>& x);

	/**
	 * Walk through the circuit and verify the multiplication gates.
	 * We first generate the random elements using a common AES key that was generated by the parties,
	 * run the relevant verification algorithm and return accept/reject according to the output
	 * of the verification algorithm.
	 */
	void verificationPhase();

	bool comparingViews();

	vector<byte> generateCommonKey();
	void generatePseudoRandomElements(vector<byte> & aesKey,
			vector<FieldType> &randomElementsToFill, int numOfRandomElements);

	bool verificationBatched(FieldType *x, FieldType *randomElements,
			int numOfTriples);

	/**
	 * Walk through the circuit and reconstruct output gates.
	 */
	void outputPhase();

	~ProtocolParty();

	void batchConsistencyCheckOfShares(const vector<FieldType> &inputShares);

};

template<class FieldType>
ProtocolParty<FieldType>::ProtocolParty(int argc, char* argv[]) :
		Protocol("MPCHonestMajorityNoTriples", argc, argv) {
	string circuitFile = this->getParser().getValueByKey(arguments,
			"circuitFile");
	this->times = stoi(
			this->getParser().getValueByKey(arguments,
					"internalIterationsNumber"));
	string fieldType = this->getParser().getValueByKey(arguments, "fieldType");
	m_partyId = stoi(this->getParser().getValueByKey(arguments, "partyID"));
	int n = stoi(this->getParser().getValueByKey(arguments, "partiesNumber"));
	string outputTimerFileName = circuitFile + "Times" + to_string(m_partyId)
			+ fieldType + ".csv";
	ProtocolTimer p(times, outputTimerFileName);

	this->protocolTimer = new ProtocolTimer(times, outputTimerFileName);

	vector<string> subTaskNames { "Offline", "preparationPhase", "Online",
			"inputPhase", "ComputePhase", "VerificationPhase", "outputPhase" };
	timer = new Measurement(*this, subTaskNames);

	if (fieldType.compare("ZpMersenne") == 0) {
		field = new TemplateField<FieldType>(2147483647);
	} else if (fieldType.compare("ZpMersenne61") == 0) {
		field = new TemplateField<FieldType>(0);
	} else if (fieldType.compare("ZpKaratsuba") == 0) {
		field = new TemplateField<FieldType>(0);
	} else if (fieldType.compare("GF2E") == 0) {
		field = new TemplateField<FieldType>(8);
	} else if (fieldType.compare("Zp") == 0) {
		field = new TemplateField<FieldType>(2147483647);
	}

	N = n;
	T = n / 2 - 1;
	this->inputsFile = this->getParser().getValueByKey(arguments, "inputFile");
	this->outputFile = this->getParser().getValueByKey(arguments, "outputFile");
	if (n % 2 > 0) {
		T++;
	}

	s = to_string(m_partyId);
	circuit.readCircuit(circuitFile.c_str());
	circuit.reArrangeCircuit();
	M = circuit.getNrOfGates();
	numOfInputGates = circuit.getNrOfInputGates();
	numOfOutputGates = circuit.getNrOfOutputGates();
	myInputs.resize(numOfInputGates);
	counter = 0;

	string partiesFile = this->getParser().getValueByKey(arguments,
			"partiesFile");

	logcat = "pp";
	init_log("protocol_party.log", "./logs", 700, logcat.c_str());
	logcat_tim = logcat + ".tim";
	logcat_out = logcat + ".out";

	int errcode = 0;
	if (0 != (errcode = pthread_mutex_init(&m_qlock, NULL))) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_mutex_init() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
		exit(__LINE__);
	}
	if (0 != (errcode = pthread_mutex_init(&m_elock, NULL))) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_mutex_init() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
		exit(__LINE__);
	}
	if (0 != (errcode = pthread_cond_init(&m_comm_e, NULL))) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_cond_init() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
		exit(__LINE__);
	}

	size_t pid = 0;
	m_parties.resize(n);

	comm_client::cc_args_t cc_args;
	cc_args.logcat = "pp.hm.nt";
	m_cc = comm_client_factory::create_comm_client(
			comm_client_factory::cc_tcp_mesh, &cc_args);
	if (0 != m_cc->start(m_partyId, n, partiesFile.c_str(), this)) {
		log4cpp::Category::getInstance(logcat).error(
				"%s: comm client start failed; run failure.", __FUNCTION__);
		exit(__LINE__);
	}

	wait_for_peer_connections();

	readMyInputs();

	auto t1 = high_resolution_clock::now();
	initializationPhase(/*matrix_him, matrix_vand, m*/);

	auto t2 = high_resolution_clock::now();

	auto duration = duration_cast < milliseconds > (t2 - t1).count();
	log4cpp::Category::getInstance(logcat_tim).info(
			"%s: time in milliseconds initializationPhase: %lu.", __FUNCTION__,
			(size_t) duration);
}

template<class FieldType>
void ProtocolParty<FieldType>::readMyInputs() {

	ifstream myfile;
	long input;
	int i = 0;
	myfile.open(inputsFile);
	do {
		myfile >> input;
		myInputs[i] = input;
		i++;
	} while (!(myfile.eof()));
	myfile.close();

}

template<class FieldType>
void ProtocolParty<FieldType>::run() {

	for (iteration = 0; iteration < times; iteration++) {

		auto t1start = high_resolution_clock::now();
		timer->startSubTask("Offline", iteration);
		runOffline();
		timer->endSubTask("Offline", iteration);
		timer->startSubTask("Online", iteration);
		runOnline();
		timer->endSubTask("Online", iteration);

		auto t2end = high_resolution_clock::now();
		auto duration = duration_cast < milliseconds
				> (t2end - t1start).count();
		protocolTimer->totalTimeArr[iteration] = duration;
		log4cpp::Category::getInstance(logcat_tim).info(
				"%s: time in milliseconds for protocol: %lu.", __FUNCTION__,
				(size_t) duration);
	}
}

template<class FieldType>
void ProtocolParty<FieldType>::runOffline() {
	auto t1 = high_resolution_clock::now();
	timer->startSubTask("preparationPhase", iteration);
	if (preparationPhase() == false) {
		log4cpp::Category::getInstance(logcat).warn("%s: cheat; aborting.",
				__FUNCTION__);
		return;
	} else {
		log4cpp::Category::getInstance(logcat).notice(
				"%s: no cheat; Preparation Phase finished.", __FUNCTION__);
	}
	timer->endSubTask("preparationPhase", iteration);
	auto t2 = high_resolution_clock::now();

	auto duration = duration_cast < milliseconds > (t2 - t1).count();
	protocolTimer->preparationPhaseArr[iteration] = duration;
	log4cpp::Category::getInstance(logcat_tim).info(
			"%s: time in milliseconds for preparationPhase: %lu.", __FUNCTION__,
			(size_t) duration);
}

template<class FieldType>
void ProtocolParty<FieldType>::runOnline() {

	auto t1 = high_resolution_clock::now();
	timer->startSubTask("inputPhase", iteration);
	inputPhase();
	timer->endSubTask("inputPhase", iteration);
	auto t2 = high_resolution_clock::now();

	auto duration = duration_cast < milliseconds > (t2 - t1).count();
	protocolTimer->inputPreparationArr[iteration] = duration;
	log4cpp::Category::getInstance(logcat_tim).info(
			"%s: time in milliseconds for inputPhase: %lu.", __FUNCTION__,
			(size_t) duration);

	t1 = high_resolution_clock::now();
	timer->startSubTask("ComputePhase", iteration);
	computationPhase(m);
	timer->endSubTask("ComputePhase", iteration);
	t2 = high_resolution_clock::now();

	duration = duration_cast < milliseconds > (t2 - t1).count();
	protocolTimer->computationPhaseArr[iteration] = duration;
	log4cpp::Category::getInstance(logcat_tim).info(
			"%s: time in milliseconds for computationPhase: %lu.", __FUNCTION__,
			(size_t) duration);

	t1 = high_resolution_clock::now();
	timer->startSubTask("VerificationPhase", iteration);
	verificationPhase();
	timer->endSubTask("VerificationPhase", iteration);
	t2 = high_resolution_clock::now();
	duration = duration_cast < milliseconds > (t2 - t1).count();
	protocolTimer->verificationPhaseArr[iteration] = duration;
	log4cpp::Category::getInstance(logcat_tim).info(
			"%s: time in milliseconds for verificationPhase: %lu.",
			__FUNCTION__, (size_t) duration);

	t1 = high_resolution_clock::now();
	timer->startSubTask("outputPhase", iteration);
	outputPhase();
	timer->endSubTask("outputPhase", iteration);
	t2 = high_resolution_clock::now();

	duration = duration_cast < milliseconds > (t2 - t1).count();
	protocolTimer->outputPhaseArr[iteration] = duration;
	log4cpp::Category::getInstance(logcat_tim).info(
			"%s: time in milliseconds for outputPhase: %lu.", __FUNCTION__,
			(size_t) duration);

}

template<class FieldType>
void ProtocolParty<FieldType>::computationPhase(HIM<FieldType> &m) {
	int count = 0;
	int countNumMult = 0;
	int countNumMultForThisLayer = 0;

	int numOfLayers = circuit.getLayers().size();
	for (int i = 0; i < numOfLayers - 1; i++) {

		currentCirciutLayer = i;
		count = processNotMult();

		countNumMultForThisLayer = processMultiplications(countNumMult); //send the index of the current mult gate
		countNumMult += countNumMultForThisLayer;
		;
		count += countNumMultForThisLayer;

	}
}

/**
 * the function implements the second step of Input Phase:
 * the party broadcasts for each input gate the difference between
 * the random secret and the actual input value.
 * @param diff
 */
template<class FieldType>
void ProtocolParty<FieldType>::inputPhase() {
	int robin = 0;

	// the number of random double sharings we need altogether
	vector<FieldType> x1(N), y1(N);
	vector<vector<FieldType>> sendBufsElements(N);
	vector < vector < byte >> sendBufsBytes(N);
	vector < vector < byte >> recBufBytes(N);
	vector<vector<FieldType>> recBufElements(N);

	int index = 0;
	vector<int> sizes(N);

	// prepare the shares for the input
	for (int k = 0; k < numOfInputGates; k++) {
		if (circuit.getGates()[k].gateType == INPUT) {
			//get the expected sized from the other parties
			sizes[circuit.getGates()[k].party]++;

			if (circuit.getGates()[k].party == m_partyId) {
				auto input = myInputs[index];
				index++;
				log4cpp::Category::getInstance(logcat).debug("%s: input = %u",
						__FUNCTION__, input);

				// the value of a_0 is the input of the party.
				x1[0] = field->GetElement(input);

				// generate random degree-T polynomial
				for (int i = 1; i < T + 1; i++) {
					// A random field element, uniform distribution
					x1[i] = field->Random();

				}

				matrix_vand.MatrixMult(x1, y1, T + 1); // eval poly at alpha-positions predefined to be alpha_i = i

				// prepare shares to be sent
				for (int i = 0; i < N; i++) {
					sendBufsElements[i].push_back(y1[i]);

				}
			}
		}
	}

	int fieldByteSize = field->getElementSizeInBytes();
	for (int i = 0; i < N; i++) {
		sendBufsBytes[i].resize(sendBufsElements[i].size() * fieldByteSize);
		recBufBytes[i].resize(sizes[i] * fieldByteSize);
		for (int j = 0; j < sendBufsElements[i].size(); j++) {
			field->elementToBytes(sendBufsBytes[i].data() + (j * fieldByteSize),
					sendBufsElements[i][j]);
		}
	}

	roundFunctionSync(sendBufsBytes, recBufBytes, 10);

	//turn the bytes to elements
	for (int i = 0; i < N; i++) {
		recBufElements[i].resize((recBufBytes[i].size()) / fieldByteSize);
		for (int j = 0; j < recBufElements[i].size(); j++) {
			recBufElements[i][j] = field->bytesToElement(
					recBufBytes[i].data() + (j * fieldByteSize));
		}
	}

	vector<int> counters(N);

	for (int i = 0; i < N; i++) {
		counters[i] = 0;
	}

	vector<FieldType> inputShares(circuit.getNrOfInputGates());

	for (int k = 0; k < numOfInputGates; k++) {
		if (circuit.getGates()[k].gateType == INPUT) {
			auto share =
					recBufElements[circuit.getGates()[k].party][counters[circuit.getGates()[k].party]];
			counters[circuit.getGates()[k].party] += 1;
			gateShareArr[circuit.getGates()[k].output * 2] = share; // set the share sent from the party owning the input
			inputShares[k] = share;

		}
	}

	inputVerification(inputShares);

	//get a random share r;

	//first generate numOfTriples random shares
	generateRandomSharesWithCheck(1, bigR);

	//set this random share to an entire array so we can use the semi honest multiplication
	vector<FieldType> resultMult(numOfInputGates, bigR[0]);

	//run the semi honest multiplication to get the second part of each share
	DNHonestMultiplication(inputShares.data(), resultMult.data(), resultMult,
			numOfInputGates);

	//set the resulted multiplication to the array of shares

	for (int k = 0; k < numOfInputGates; k++) {
		if (circuit.getGates()[k].gateType == INPUT) {
			//set the second part of the share
			gateShareArr[circuit.getGates()[k].output * 2 + 1] = resultMult[k];

		}
	}

}

template<class FieldType>
void ProtocolParty<FieldType>::inputVerification(
		vector<FieldType> &inputShares) {

	batchConsistencyCheckOfShares(inputShares);

}

template<class FieldType>
void ProtocolParty<FieldType>::batchConsistencyCheckOfShares(
		const vector<FieldType> &inputShares) { //first generate the common aes key

	auto key = generateCommonKey();

	//print key
	for (int i = 0; i < key.size(); i++)
		log4cpp::Category::getInstance(logcat).info(
				"%s: key[%d] for party[%d] is: %d.", __FUNCTION__, i, m_partyId,
				key[i]);

	//calc the number of times we need to run the verification -- ceiling
	int iterations = (5 + field->getElementSizeInBytes() - 1)
			/ field->getElementSizeInBytes();

	vector<FieldType> randomElements(inputShares.size() * iterations);
	generatePseudoRandomElements(key, randomElements, inputShares.size());

	for (int j = 0; j < iterations; j++) {
		vector<FieldType> r(1); //vector holding the random shares generated
		vector<FieldType> v(1);
		vector<FieldType> secret(1);

		getRandomShares(1, r);

		for (int i = 0; i < inputShares.size(); i++)
			v[0] += randomElements[i + j * inputShares.size()] * inputShares[i];

		v[0] += r[0];

		//if all the the parties share lie on the same polynomial this will not abort
		openShare(1, v, secret);
	}
}

template<class FieldType>
void ProtocolParty<FieldType>::generateRandomSharesWithCheck(int numOfRandoms,
		vector<FieldType>& randomElementsToFill) {

	getRandomShares(numOfRandoms, randomElementsToFill);

	batchConsistencyCheckOfShares(randomElementsToFill);

}

template<class FieldType>
void ProtocolParty<FieldType>::generateRandomShares(int numOfRandoms,
		vector<FieldType> &randomElementsToFill) {
	int index = 0;
	vector < vector < byte >> recBufsBytes(N);
	int robin = 0;
	int no_random = numOfRandoms;

	vector<FieldType> x1(N), y1(N), x2(N), y2(N), t1(N), r1(N), t2(N), r2(N);
	vector<vector<FieldType>> sendBufsElements(N);
	vector < vector < byte >> sendBufsBytes(N);

	// the number of buckets (each bucket requires one double-sharing
	// from each party and gives N-2T random double-sharings)
	int no_buckets = (no_random / (N - T)) + 1;

	//sharingBufTElements.resize(no_buckets*(N-2*T)); // my shares of the double-sharings
	//sharingBuf2TElements.resize(no_buckets*(N-2*T)); // my shares of the double-sharings

	//maybe add some elements if a partial bucket is needed
	randomElementsToFill.resize(no_buckets * (N - T));

	for (int i = 0; i < N; i++) {
		sendBufsElements[i].resize(no_buckets);
		sendBufsBytes[i].resize(no_buckets * field->getElementSizeInBytes());
		recBufsBytes[i].resize(no_buckets * field->getElementSizeInBytes());
	}

	/**
	 *  generate random sharings.
	 *  first degree t.
	 *
	 */
	for (int k = 0; k < no_buckets; k++) {
		// generate random degree-T polynomial
		for (int i = 0; i < T + 1; i++) {
			// A random field element, uniform distribution, note that x1[0] is the secret which is also random
			x1[i] = field->Random();

		}

		matrix_vand.MatrixMult(x1, y1, T + 1); // eval poly at alpha-positions

		// prepare shares to be sent
		for (int i = 0; i < N; i++) {
			sendBufsElements[i][k] = y1[i];

		}
	}

	log4cpp::Category::getInstance(logcat).debug("%s: N=%d; T=%d.",
			__FUNCTION__, N, T);

	int fieldByteSize = field->getElementSizeInBytes();
	for (int i = 0; i < N; i++) {
//        for(int j=0; j<sendBufsElements[i].size();j++) {
//            field->elementToBytes(sendBufsBytes[i].data() + (j * fieldByteSize), sendBufsElements[i][j]);
//        }

		field->elementVectorToByteVector(sendBufsElements[i], sendBufsBytes[i]);
	}

	roundFunctionSync(sendBufsBytes, recBufsBytes, 4);

	for (int k = 0; k < no_buckets; k++) {
		for (int i = 0; i < N; i++) {
			t1[i] = field->bytesToElement(
					recBufsBytes[i].data() + (k * fieldByteSize));

		}
		matrix_vand_transpose.MatrixMult(t1, r1, N - T);

		//copy the resulting vector to the array of randoms
		for (int i = 0; i < N - T; i++) {

			randomElementsToFill[index] = r1[i];
			index++;

		}
	}
}
template<class FieldType>
void ProtocolParty<FieldType>::getRandomShares(int numOfRandoms,
		vector<FieldType> &randomElementsToFill) {

	randomElementsToFill.assign(randomSharesArray.begin() + randomSharesOffset,
			randomSharesArray.begin() + randomSharesOffset + numOfRandoms);

	randomSharesOffset += numOfRandoms;

}

template<class FieldType>
void ProtocolParty<FieldType>::generateRandom2TAndTShares(int numOfRandomPairs,
		vector<FieldType>& randomElementsToFill) {

	int index = 0;
	vector < vector < byte >> recBufsBytes(N);
	int robin = 0;
	int no_random = numOfRandomPairs;

	vector<FieldType> x1(N), y1(N), x2(N), y2(N), t1(N), r1(N), t2(N), r2(N);
	;

	vector<vector<FieldType>> sendBufsElements(N);
	vector < vector < byte >> sendBufsBytes(N);

	// the number of buckets (each bucket requires one double-sharing
	// from each party and gives N-2T random double-sharings)
	int no_buckets = (no_random / (N - T)) + 1;

	//sharingBufTElements.resize(no_buckets*(N-2*T)); // my shares of the double-sharings
	//sharingBuf2TElements.resize(no_buckets*(N-2*T)); // my shares of the double-sharings

	//maybe add some elements if a partial bucket is needed
	randomElementsToFill.resize(no_buckets * (N - T) * 2);
	vector<FieldType> randomElementsOnlyTshares(no_buckets * (N - T));

	for (int i = 0; i < N; i++) {
		sendBufsElements[i].resize(no_buckets * 2);
		sendBufsBytes[i].resize(
				no_buckets * field->getElementSizeInBytes() * 2);
		recBufsBytes[i].resize(no_buckets * field->getElementSizeInBytes() * 2);
	}

	/**
	 *  generate random sharings.
	 *  first degree t.
	 *
	 */
	for (int k = 0; k < no_buckets; k++) {
		// generate random degree-T polynomial
		for (int i = 0; i < T + 1; i++) {
			// A random field element, uniform distribution, note that x1[0] is the secret which is also random
			x1[i] = field->Random();

		}

		matrix_vand.MatrixMult(x1, y1, T + 1); // eval poly at alpha-positions

		x2[0] = x1[0];
		// generate random degree-T polynomial
		for (int i = 1; i < 2 * T + 1; i++) {
			// A random field element, uniform distribution, note that x1[0] is the secret which is also random
			x2[i] = field->Random();

		}

		matrix_vand.MatrixMult(x2, y2, 2 * T + 1);

		// prepare shares to be sent
		for (int i = 0; i < N; i++) {
			sendBufsElements[i][2 * k] = y1[i];
			sendBufsElements[i][2 * k + 1] = y2[i];

		}
	}

	log4cpp::Category::getInstance(logcat).debug("%s: N=%d; T=%d.",
			__FUNCTION__, N, T);

	int fieldByteSize = field->getElementSizeInBytes();
	for (int i = 0; i < N; i++) {
//        for(int j=0; j<sendBufsElements[i].size();j++) {
//            field->elementToBytes(sendBufsBytes[i].data() + (j * fieldByteSize), sendBufsElements[i][j]);
//        }

		field->elementVectorToByteVector(sendBufsElements[i], sendBufsBytes[i]);
	}

	roundFunctionSync(sendBufsBytes, recBufsBytes, 4);

	for (int k = 0; k < no_buckets; k++) {
		for (int i = 0; i < N; i++) {
			t1[i] = field->bytesToElement(
					recBufsBytes[i].data() + (2 * k * fieldByteSize));
			t2[i] = field->bytesToElement(
					recBufsBytes[i].data() + ((2 * k + 1) * fieldByteSize));

		}
		matrix_vand_transpose.MatrixMult(t1, r1, N - T);
		matrix_vand_transpose.MatrixMult(t2, r2, N - T);

		//copy the resulting vector to the array of randoms
		for (int i = 0; i < (N - T); i++) {

			randomElementsToFill[index * 2] = r1[i];
			randomElementsToFill[index * 2 + 1] = r2[i];
			index++;

		}
	}

	//check validity of the t-shares. 2t-shares do not have to be checked
	//copy the t-shares for checking

	for (int i = 0; i < randomElementsOnlyTshares.size(); i++) {

		randomElementsOnlyTshares[i] = randomElementsToFill[2 * i];
	}

	batchConsistencyCheckOfShares(randomElementsOnlyTshares);

}

/**
 * some global variables are initialized
 * @param GateValueArr
 * @param GateShareArr
 * @param matrix_him
 * @param matrix_vand
 * @param alpha
 */
template<class FieldType>
void ProtocolParty<FieldType>::initializationPhase() {
	bigR.resize(1);
	beta.resize(1);
	y_for_interpolate.resize(N);
	gateShareArr.resize((M - circuit.getNrOfOutputGates()) * 2); // my share of the gate (for all gates)
	alpha.resize(N); // N distinct non-zero field elements
	vector<FieldType> alpha1(N - T);
	vector<FieldType> alpha2(T);

	beta[0] = field->GetElement(0); // zero of the field
	matrix_for_interpolate.allocate(1, N, field);

	matrix_him.allocate(N, N, field);
	matrix_vand.allocate(N, N, field);
	matrix_vand_transpose.allocate(N, N, field);
	m.allocate(T, N - T, field);

	// Compute Vandermonde matrix VDM[i,k] = alpha[i]^k
	matrix_vand.InitVDM();
	matrix_vand_transpose.InitVDMTranspose();

	// Prepare an N-by-N hyper-invertible matrix
	matrix_him.InitHIM();

	// N distinct non-zero field elements
	for (int i = 0; i < N; i++) {
		alpha[i] = field->GetElement(i + 1);
	}

	for (int i = 0; i < N - T; i++) {
		alpha1[i] = alpha[i];
	}
	for (int i = N - T; i < N; i++) {
		alpha2[i - (N - T)] = alpha[i];
	}

	m.InitHIMByVectors(alpha1, alpha2);

	matrix_for_interpolate.InitHIMByVectors(alpha, beta);

	vector<FieldType> alpha_until_t(T + 1);
	vector<FieldType> alpha_from_t(N - 1 - T);

	// Interpolate first d+1 positions of (alpha,x)
	matrix_for_t.allocate(N - 1 - T, T + 1, field); // slices, only positions from 0..d
	//matrix_for_t.InitHIMByVectors(alpha_until_t, alpha_from_t);
	matrix_for_t.InitHIMVectorAndsizes(alpha, T + 1, N - T - 1);

	vector<FieldType> alpha_until_2t(2 * T + 1);
	vector<FieldType> alpha_from_2t(N - 1 - 2 * T);

	// Interpolate first d+1 positions of (alpha,x)
	matrix_for_2t.allocate(N - 1 - 2 * T, 2 * T + 1, field); // slices, only positions from 0..d
	//matrix_for_2t.InitHIMByVectors(alpha_until_2t, alpha_from_2t);
	matrix_for_2t.InitHIMVectorAndsizes(alpha, 2 * T + 1, N - (2 * T + 1));
}

template<class FieldType>
bool ProtocolParty<FieldType>::preparationPhase() {
	int iterations = (5 + field->getElementSizeInBytes() - 1)
			/ field->getElementSizeInBytes();
	int keysize = 16 / field->getElementSizeInBytes() + 1;

	int numOfRandomShares = 5 * keysize + 3 * iterations + 1;
	randomSharesArray.resize(numOfRandomShares);

	randomSharesOffset = 0;
	//generate enough random shares for the AES key
	generateRandomShares(numOfRandomShares, randomSharesArray);

	//run offline for all the future multiplications including the multiplication of the protocol

	offset = 0;
	offlineDNForMultiplication(
			(circuit.getNrOfInputGates()
					+ circuit.getNrOfMultiplicationGates() * 2 + 1)
					* iterations);

	return true;
}

/**
 * Check whether given points lie on polynomial of degree d. This check is performed by interpolating x on
 * the first d + 1 positions of α and check the remaining positions.
 */
template<class FieldType>
bool ProtocolParty<FieldType>::checkConsistency(vector<FieldType>& x, int d) {
	if (d == T) {
		vector<FieldType> y(N - 1 - d); // the result of multiplication
		vector<FieldType> x_until_t(T + 1);

		for (int i = 0; i < T + 1; i++) {
			x_until_t[i] = x[i];
		}

		matrix_for_t.MatrixMult(x_until_t, y);

		// compare that the result is equal to the according positions in x
		for (int i = 0; i < N - d - 1; i++)   // n-d-2 or n-d-1 ??
				{
			if ((y[i]) != (x[d + 1 + i])) {
				return false;
			}
		}
		return true;
	} else if (d == 2 * T) {
		vector<FieldType> y(N - 1 - d); // the result of multiplication

		vector<FieldType> x_until_2t(2 * T + 1);

		for (int i = 0; i < 2 * T + 1; i++) {
			x_until_2t[i] = x[i];
		}

		matrix_for_2t.MatrixMult(x_until_2t, y);

		// compare that the result is equal to the according positions in x
		for (int i = 0; i < N - d - 1; i++)   // n-d-2 or n-d-1 ??
				{
			if ((y[i]) != (x[d + 1 + i])) {
				return false;
			}
		}
		return true;

	} else {
		vector<FieldType> alpha_until_d(d + 1);
		vector<FieldType> alpha_from_d(N - 1 - d);
		vector<FieldType> x_until_d(d + 1);
		vector<FieldType> y(N - 1 - d); // the result of multiplication

		for (int i = 0; i < d + 1; i++) {
			alpha_until_d[i] = alpha[i];
			x_until_d[i] = x[i];
		}
		for (int i = d + 1; i < N; i++) {
			alpha_from_d[i - (d + 1)] = alpha[i];
		}
		// Interpolate first d+1 positions of (alpha,x)
		HIM<FieldType> matrix(N - 1 - d, d + 1, field); // slices, only positions from 0..d
		matrix.InitHIMByVectors(alpha_until_d, alpha_from_d);
		matrix.MatrixMult(x_until_d, y);

		// compare that the result is equal to the according positions in x
		for (int i = 0; i < N - d - 1; i++)   // n-d-2 or n-d-1 ??
				{
			if (y[i] != x[d + 1 + i]) {
				return false;
			}
		}
		return true;
	}
	return true;
}

// Interpolate polynomial at position Zero
template<class FieldType>
FieldType ProtocolParty<FieldType>::interpolate(vector<FieldType>& x) {
	//vector<FieldType> y(N); // result of interpolate
	matrix_for_interpolate.MatrixMult(x, y_for_interpolate);
	return y_for_interpolate[0];
}

template<class FieldType>
FieldType ProtocolParty<FieldType>::reconstructShare(vector<FieldType>& x,
		int d) {

	if (!checkConsistency(x, d)) {
		// someone cheated!
		log4cpp::Category::getInstance(logcat).error("%s: cheat deteceted.",
				__FUNCTION__);
		exit(__LINE__);
	} else
		return interpolate(x);
}

template<class FieldType>
int ProtocolParty<FieldType>::processNotMult() {
	int count = 0;
	for (int k = circuit.getLayers()[currentCirciutLayer];
			k < circuit.getLayers()[currentCirciutLayer + 1]; k++) {

		// add gate
		if (circuit.getGates()[k].gateType == ADD) {
			gateShareArr[circuit.getGates()[k].output * 2] =
					gateShareArr[circuit.getGates()[k].input1 * 2]
							+ gateShareArr[circuit.getGates()[k].input2 * 2];
			gateShareArr[circuit.getGates()[k].output * 2 + 1] =
					gateShareArr[circuit.getGates()[k].input1 * 2 + 1]
							+ gateShareArr[circuit.getGates()[k].input2 * 2 + 1];
			count++;
		}

		else if (circuit.getGates()[k].gateType == SUB)   //sub gate
				{
			gateShareArr[circuit.getGates()[k].output * 2] =
					gateShareArr[circuit.getGates()[k].input1 * 2]
							- gateShareArr[circuit.getGates()[k].input2 * 2];
			gateShareArr[circuit.getGates()[k].output * 2 + 1] =
					gateShareArr[circuit.getGates()[k].input1 * 2 + 1]
							- gateShareArr[circuit.getGates()[k].input2 * 2 + 1];
			count++;
		} else if (circuit.getGates()[k].gateType == SCALAR) {
			long scalar(circuit.getGates()[k].input2);
			FieldType e = field->GetElement(scalar);
			gateShareArr[circuit.getGates()[k].output * 2] =
					gateShareArr[circuit.getGates()[k].input1 * 2] * e;
			gateShareArr[circuit.getGates()[k].output * 2 + 1] =
					gateShareArr[circuit.getGates()[k].input1 * 2 + 1] * e;

			count++;
		} else if (circuit.getGates()[k].gateType == SCALAR_ADD) {
			long scalar(circuit.getGates()[k].input2);
			FieldType e = field->GetElement(scalar);
			gateShareArr[circuit.getGates()[k].output * 2] =
					gateShareArr[circuit.getGates()[k].input1 * 2] + e;
			gateShareArr[circuit.getGates()[k].output * 2 + 1] =
					gateShareArr[circuit.getGates()[k].input1 * 2 + 1] + e;

			count++;
		}

	}
	return count;

}

/**
 * the Function process all multiplications which are ready.
 * @return the number of processed gates.
 */
template<class FieldType>
int ProtocolParty<FieldType>::processMultiplications(int lastMultGate) {

	return processMultDN(lastMultGate);

}

template<class FieldType>
int ProtocolParty<FieldType>::processMultDN(int indexInRandomArray) {

	int index = 0;
	int fieldByteSize = field->getElementSizeInBytes();
	int maxNumberOfLayerMult = circuit.getLayers()[currentCirciutLayer + 1]
			- circuit.getLayers()[currentCirciutLayer];
	vector<FieldType> xyMinusRShares(maxNumberOfLayerMult * 2); //hold both in the same vector to send in one batch

	vector<FieldType> xyMinusR; //hold both in the same vector to send in one batch
	vector<byte> xyMinusRBytes;

	vector < vector < byte >> recBufsBytes(N);
	vector < vector < byte >> sendBufsBytes(N);
	vector<vector<FieldType>> sendBufsElements(N);

	//generate the shares for x+a and y+b. do it in the same array to send once
	for (int k = circuit.getLayers()[currentCirciutLayer];
			k < circuit.getLayers()[currentCirciutLayer + 1]; k++) //go over only the logit gates
			{
		auto gate = circuit.getGates()[k];

		if (gate.gateType == MULT) {

			//compute the share of xy-r
			xyMinusRShares[index * 2] = gateShareArr[gate.input1 * 2]
					* gateShareArr[gate.input2 * 2]
					- randomTAnd2TShares[2 * indexInRandomArray + 1];

			indexInRandomArray++;

			//compute the share of xy-r
			xyMinusRShares[index * 2 + 1] = gateShareArr[gate.input1 * 2 + 1]
					* gateShareArr[gate.input2 * 2]
					- randomTAnd2TShares[2 * indexInRandomArray + 1];

			indexInRandomArray++;

			index++;
		}
	}

	if (index == 0)
		return 0;

	//set the acctual number of mult gate proccessed in this layer
	int acctualNumOfMultGates = index;
	int numOfElementsForParties = acctualNumOfMultGates / N;
	int indexForDecreasingSize = acctualNumOfMultGates
			- numOfElementsForParties * N;

	int counter = 0;
	int currentNumOfElements;
	for (int i = 0; i < N; i++) {

		currentNumOfElements = numOfElementsForParties;
		if (i < indexForDecreasingSize)
			currentNumOfElements++;

		//fill the send buf according to the number of elements to send to each party
		sendBufsElements[i].resize(currentNumOfElements * 2);
		sendBufsBytes[i].resize(currentNumOfElements * fieldByteSize * 2);
		for (int j = 0; j < currentNumOfElements * 2; j++) {

			sendBufsElements[i][j] = xyMinusRShares[counter];
			counter++;

		}
		field->elementVectorToByteVector(sendBufsElements[i], sendBufsBytes[i]);

	}

	//resize the recbuf array.
	int myNumOfElementsToExpect = numOfElementsForParties;
	if (m_partyId < indexForDecreasingSize) {
		myNumOfElementsToExpect = numOfElementsForParties + 1;
	}
	for (int i = 0; i < N; i++) {

		recBufsBytes[i].resize(myNumOfElementsToExpect * fieldByteSize * 2);
	}

	roundFunctionSync(sendBufsBytes, recBufsBytes, 20);

	xyMinusR.resize(myNumOfElementsToExpect * 2);
	xyMinusRBytes.resize(myNumOfElementsToExpect * fieldByteSize * 2);

	//reconstruct the shares that I am responsible of recieved from the other parties
	vector<FieldType> xyMinurAllShares(N);

	for (int k = 0; k < myNumOfElementsToExpect * 2; k++) //go over only the logit gates
			{
		for (int i = 0; i < N; i++) {

			xyMinurAllShares[i] = field->bytesToElement(
					recBufsBytes[i].data() + (k * fieldByteSize));
		}

		// reconstruct the shares by P0
		xyMinusR[k] = interpolate(xyMinurAllShares);

	}

	field->elementVectorToByteVector(xyMinusR, xyMinusRBytes);

	//prepare the send buffers
	for (int i = 0; i < N; i++) {
		sendBufsBytes[i] = xyMinusRBytes;
	}

	for (int i = 0; i < N; i++) {

		currentNumOfElements = numOfElementsForParties;
		if (i < indexForDecreasingSize)
			currentNumOfElements++;

		recBufsBytes[i].resize(currentNumOfElements * fieldByteSize * 2);

	}

	roundFunctionSync(sendBufsBytes, recBufsBytes, 21);

	xyMinusR.resize(acctualNumOfMultGates * 2);
	counter = 0;

	for (int i = 0; i < N; i++) {

		currentNumOfElements = numOfElementsForParties;
		if (i < indexForDecreasingSize)
			currentNumOfElements++;

		//fill the send buf according to the number of elements to send to each party
		for (int j = 0; j < currentNumOfElements * 2; j++) {

			xyMinusR[counter] = field->bytesToElement(
					recBufsBytes[i].data() + (j * fieldByteSize));
			counter++;

		}

	}

	indexInRandomArray -= 2 * index;
	index = 0;

	//after the xPlusAAndYPlusB array is filled, we are ready to fill the output of the mult gates
	for (int k = circuit.getLayers()[currentCirciutLayer];
			k < circuit.getLayers()[currentCirciutLayer + 1]; k++) //go over only the logit gates
			{
		auto gate = circuit.getGates()[k];

		if (gate.gateType == MULT) {

			gateShareArr[gate.output * 2] = randomTAnd2TShares[2
					* indexInRandomArray] + xyMinusR[index * 2];
			indexInRandomArray++;
			gateShareArr[gate.output * 2 + 1] = randomTAnd2TShares[2
					* indexInRandomArray] + xyMinusR[index * 2 + 1];

			index++;
			indexInRandomArray++;

		}
	}

	return index;
}

template<class FieldType>
void ProtocolParty<FieldType>::DNHonestMultiplication(FieldType *a,
		FieldType *b, vector<FieldType> &cToFill, int numOfTrupples) {

	int fieldByteSize = field->getElementSizeInBytes();
	vector<FieldType> xyMinusRShares(numOfTrupples); //hold both in the same vector to send in one batch
	vector<byte> xyMinusRSharesBytes(numOfTrupples * fieldByteSize); //hold both in the same vector to send in one batch

	vector<FieldType> xyMinusR; //hold both in the same vector to send in one batch
	vector<byte> xyMinusRBytes;

	vector < vector < byte >> recBufsBytes;

	//generate the shares for x+a and y+b. do it in the same array to send once
	for (int k = 0; k < numOfTrupples; k++)   //go over only the logit gates
			{
		//compute the share of xy-r
		xyMinusRShares[k] = a[k] * b[k]
				- randomTAnd2TShares[offset + 2 * k + 1];

	}

	//set the acctual number of mult gate proccessed in this layer
	xyMinusRSharesBytes.resize(numOfTrupples * fieldByteSize);
	xyMinusR.resize(numOfTrupples);
	xyMinusRBytes.resize(numOfTrupples * fieldByteSize);

//    for(int j=0; j<xyMinusRShares.size();j++) {
//        field->elementToBytes(xyMinusRSharesBytes.data() + (j * fieldByteSize), xyMinusRShares[j]);
//    }

	field->elementVectorToByteVector(xyMinusRShares, xyMinusRSharesBytes);

	if (m_partyId == 0) {

		//just party 1 needs the recbuf
		recBufsBytes.resize(N);

		for (int i = 0; i < N; i++) {
			recBufsBytes[i].resize(numOfTrupples * fieldByteSize);
		}

		//receive the shares from all the other parties
		roundFunctionSyncForP1(xyMinusRSharesBytes, recBufsBytes);
	} else {   //since I am not party 1 parties[0]->getID()=1

		//send the shares to p1
		//parties[0]->getChannel()->write(xyMinusRSharesBytes.data(), xyMinusRSharesBytes.size());

		m_cc->send(0, xyMinusRSharesBytes.data(), xyMinusRSharesBytes.size());

	}

	//reconstruct the shares recieved from the other parties
	if (m_partyId == 0) {

		vector<FieldType> xyMinurAllShares(N);

		for (int k = 0; k < numOfTrupples; k++)   //go over only the logit gates
				{
			for (int i = 0; i < N; i++) {

				xyMinurAllShares[i] = field->bytesToElement(
						recBufsBytes[i].data() + (k * fieldByteSize));
			}
			/*for (int i = T*2; i < N; i++) {

			 xyMinurAllShares[i] = *(field->GetZero());
			 }*/

			// reconstruct the shares by P0
			xyMinusR[k] = interpolate(xyMinurAllShares);

			//convert to bytes
			//field->elementToBytes(xyMinusRBytes.data() + (k * fieldByteSize), xyMinusR[k]);

		}

		field->elementVectorToByteVector(xyMinusR, xyMinusRBytes);

		//send the reconstructed vector to all the other parties
		for (size_t i = 1; i < m_parties.size(); ++i) {
			m_cc->send(i, xyMinusRBytes.data(), xyMinusRBytes.size());
		}
	}

	else {            //each party get the xy-r reconstruced vector from party 1

		bool party_0_data_ready = false;
		do {
			process_network_events();
			party_0_data_ready =
					(m_parties[0].data.size() >= xyMinusRBytes.size()) ?
							true : false;
		} while (!party_0_data_ready);

		memcpy(xyMinusRBytes.data(), m_parties[0].data.data(),
				xyMinusRBytes.size());
		m_parties[0].data.erase(m_parties[0].data.begin(),
				m_parties[0].data.begin() + xyMinusRBytes.size());
	}

	if (m_partyId != 0) {

		for (int i = 0; i < numOfTrupples; i++) {

			xyMinusR[i] = field->bytesToElement(
					xyMinusRBytes.data() + (i * fieldByteSize));
		}
	}

	//add the xy-r bytes to the h vector to be hashed in the comparing views function
	h.insert(h.end(), xyMinusRBytes.begin(), xyMinusRBytes.end());

	for (int k = 0; k < numOfTrupples; k++) {
		cToFill[k] = randomTAnd2TShares[offset + 2 * k] + xyMinusR[k];
	}

	offset += numOfTrupples * 2;

}

template<class FieldType>
void ProtocolParty<FieldType>::offlineDNForMultiplication(int numOfTriples) {

	generateRandom2TAndTShares(numOfTriples, randomTAnd2TShares);

}

template<class FieldType>
void ProtocolParty<FieldType>::verificationPhase() {

	int numOfOutputGates = circuit.getNrOfOutputGates();
	int numOfInputGates = circuit.getNrOfInputGates();
	int numOfMultGates = circuit.getNrOfMultiplicationGates();
	//get the number of random elements to create

	int numOfRandomelements = numOfMultGates + numOfInputGates;
	//first generate the common aes key
	auto key = generateCommonKey();

	//print key
	for (int i = 0; i < key.size(); i++)
		log4cpp::Category::getInstance(logcat).info(
				"%s: key[%d] for party[%d] is: %d.", __FUNCTION__, i, m_partyId,
				key[i]);

	//calc the number of times we need to run the verification -- ceiling
	int iterations = (5 + field->getElementSizeInBytes() - 1)
			/ field->getElementSizeInBytes();

	//preapre x,y,z for the verification sub protocol
	vector<FieldType> neededShares(numOfRandomelements * 2);

	int index = 0;
	for (int k = 0; k < numOfInputGates; k++) {

		auto gate = circuit.getGates()[k];

		if (gate.gateType == INPUT) {
			neededShares[2 * index] = gateShareArr[gate.output * 2];
			neededShares[2 * index + 1] = gateShareArr[gate.output * 2 + 1];

			index++;
		}
	}

	for (int k = numOfInputGates - 1; k < M - numOfOutputGates + 1; k++) {

		auto gate = circuit.getGates()[k];
		if (gate.gateType == MULT) {
			neededShares[index * 2] = gateShareArr[gate.output * 2];
			neededShares[index * 2 + 1] = gateShareArr[gate.output * 2 + 1];
			index++;
		}

	}

	bool answer;

	vector<FieldType> randomElements(numOfRandomelements * iterations);
	generatePseudoRandomElements(key, randomElements,
			numOfRandomelements * iterations);

	for (int i = 0; i < iterations; i++) {

		if (i != 0) {
			//we need the accumulate h from previous phases, after the first iteration we would like to clear the
			//h in order no to compare with a bigger h than neccesary.
			h.clear();
		}

		log4cpp::Category::getInstance(logcat).info(
				"%s: verifying batch for party %d.", __FUNCTION__, m_partyId);

		//call the verification sub protocol
		answer = verificationBatched(neededShares.data(),
				randomElements.data() + numOfRandomelements * i,
				numOfMultGates + numOfInputGates);
		log4cpp::Category::getInstance(logcat).info(
				"%s: answer is %s for iteration %d.", __FUNCTION__,
				((answer) ? "true" : "false"), i);
	}
}

template<class FieldType>
bool ProtocolParty<FieldType>::comparingViews() {

	vector < vector < byte >> sendBufsBytes(N);
	vector < vector < byte >> recBufBytes(N);

	vector<byte> keyVector;
	unsigned int hashSize = 16;
	unsigned char digest[hashSize];

	//not sure this is even needed, why should the aes key be generated now and not before the entire computation
	keyVector = generateCommonKey();

	// gcm initialization vector
	unsigned char iv1[] = { 0xe0, 0xe0, 0x0f, 0x19, 0xfe, 0xd7, 0xba, 0x01,
			0x36, 0xa7, 0x97, 0xf3 };

	unsigned char *key = reinterpret_cast<unsigned char*>(keyVector.data());

	HashEncrypt hashObj = HashEncrypt(key, iv1, 12);

	hashObj.getHashedDataOnce(reinterpret_cast<unsigned char*>(h.data()),
			h.size(), digest, &hashSize);

	//put the values of the digest for each party
	for (int i = 0; i < N; i++) {

		//copy the digest to each send buf
		copy_byte_array_to_byte_vector(digest, 16, sendBufsBytes[i], 0);

		//prepare the size for the receiving digests
		recBufBytes[i].resize(16);

	}

	roundFunctionSync(sendBufsBytes, recBufBytes, 11);

	int cmp = 0;
	for (int i = 0; i < N; i++) {

		cmp += memcmp(recBufBytes[i].data(), digest, 16);
	}

	if (cmp == 0) {
		log4cpp::Category::getInstance(logcat).notice(
				"%s: all digests are the same.", __FUNCTION__);
		return true;
	} else {
		log4cpp::Category::getInstance(logcat).notice(
				"%s: comparing views failed.", __FUNCTION__);
		return false;
	}
}

template<class FieldType>
vector<byte> ProtocolParty<FieldType>::generateCommonKey() {

	int fieldByteSize = field->getElementSizeInBytes();

	//calc the number of elements needed for 128 bit AES key
	int numOfRandomShares = 16 / field->getElementSizeInBytes() + 1;
	vector<FieldType> randomSharesArray(numOfRandomShares);
	vector<FieldType> aesArray(numOfRandomShares);
	vector<byte> aesKey(numOfRandomShares * fieldByteSize);

	//generate enough random shares for the AES key

	getRandomShares(numOfRandomShares, randomSharesArray);

	openShare(numOfRandomShares, randomSharesArray, aesArray);

	//turn the aes array into bytes to get the common aes key.
	for (int i = 0; i < numOfRandomShares; i++) {

		for (int j = 0; j < numOfRandomShares; j++) {
			field->elementToBytes(aesKey.data() + (j * fieldByteSize),
					aesArray[j]);
		}
	}

	//reduce the size of the key to 16 bytes
	aesKey.resize(16);

	return aesKey;

}

template<class FieldType>
void ProtocolParty<FieldType>::openShare(int numOfRandomShares,
		vector<FieldType> &Shares, vector<FieldType> &secrets) {

	vector < vector < byte >> sendBufsBytes(N);
	vector < vector < byte >> recBufsBytes(N);

	vector<FieldType> x1(N);
	int fieldByteSize = field->getElementSizeInBytes();

	//calc the number of elements needed for 128 bit AES key

	//resize vectors
	for (int i = 0; i < N; i++) {
		sendBufsBytes[i].resize(numOfRandomShares * fieldByteSize);
		recBufsBytes[i].resize(numOfRandomShares * fieldByteSize);
	}

	//set the first sending data buffer
	for (int j = 0; j < numOfRandomShares; j++) {
		field->elementToBytes(sendBufsBytes[0].data() + (j * fieldByteSize),
				Shares[j]);
	}

	//copy the same data for all parties
	for (int i = 1; i < N; i++) {

		sendBufsBytes[i] = sendBufsBytes[0];
	}

	//call the round function to send the shares to all the users and get the other parties share
	roundFunctionSync(sendBufsBytes, recBufsBytes, 12);

	//reconstruct each set of shares to get the secret

	for (int k = 0; k < numOfRandomShares; k++) {

		//get the set of shares for each element
		for (int i = 0; i < N; i++) {

			x1[i] = field->bytesToElement(
					recBufsBytes[i].data() + (k * fieldByteSize));
		}

		secrets[k] = reconstructShare(x1, T);

	}

}

template<class FieldType>
void ProtocolParty<FieldType>::generatePseudoRandomElements(
		vector<byte> & aesKey, vector<FieldType> &randomElementsToFill,
		int numOfRandomElements) {

	int fieldSize = field->getElementSizeInBytes();
	int fieldSizeBits = field->getElementSizeInBits();
	bool isLongRandoms;
	int size;
	if (fieldSize > 4) {
		isLongRandoms = true;
		size = 8;
	} else {

		isLongRandoms = false;
		size = 4;
	}

	log4cpp::Category::getInstance(logcat).debug("%s: size is %d for party %d.",
			__FUNCTION__, size, m_partyId);

	PrgFromOpenSSLAES prg((numOfRandomElements * size / 16) + 1);
	SecretKey sk(aesKey, "aes");
	prg.setKey(sk);

	for (int i = 0; i < numOfRandomElements; i++) {

		if (isLongRandoms)
			randomElementsToFill[i] = field->GetElement(
					((unsigned long) prg.getRandom64())
							>> (64 - fieldSizeBits));
		else
			randomElementsToFill[i] = field->GetElement(prg.getRandom32());
	}

}

template<class FieldType>
bool ProtocolParty<FieldType>::verificationBatched(FieldType *neededShares,
		FieldType * randomElements, int numOfTriples) {

	vector<FieldType> u(1);
	FieldType w;
	vector<FieldType> ru(1);
	vector<FieldType> T(1);

	for (int i = 0; i < numOfTriples; i++) {

		u[0] += randomElements[i] * neededShares[i * 2];
		w += randomElements[i] * neededShares[i * 2 + 1];
	}

	//run the semi honest multiplication on u and r to get ru
	DNHonestMultiplication(u.data(), bigR.data(), ru, 1);

	T[0] = w - ru[0];

	comparingViews();

	//open [T]
	vector<FieldType> shareArr(1);
	vector<FieldType> secretArr(1);
	shareArr[0] = T[0];

	openShare(1, shareArr, secretArr);

	//check that T=0
	if (secretArr[0] != *field->GetZero()) {
		log4cpp::Category::getInstance(logcat).debug("%s: baasa.",
				__FUNCTION__);
		return false;
	} else {
		log4cpp::Category::getInstance(logcat).debug("%s: yes.", __FUNCTION__);
		return true;
	}

}

/**
 * the function Walk through the circuit and reconstruct output gates.
 * @param circuit
 * @param gateShareArr
 * @param alpha
 */
template<class FieldType>
void ProtocolParty<FieldType>::outputPhase() {
	int count = 0;
	vector<FieldType> x1(N); // vector for the shares of my outputs
	vector<vector<FieldType>> sendBufsElements(N);
	vector < vector < byte >> sendBufsBytes(N);
	vector < vector < byte >> recBufBytes(N);

	FieldType num;
	ofstream myfile;
	myfile.open(outputFile);

	for (int k = M - numOfOutputGates; k < M; k++) {
		if (circuit.getGates()[k].gateType == OUTPUT) {
			// send to party (which need this gate) your share for this gate
			sendBufsElements[circuit.getGates()[k].party].push_back(
					gateShareArr[circuit.getGates()[k].input1 * 2]);
		}
	}

	int fieldByteSize = field->getElementSizeInBytes();
	for (int i = 0; i < N; i++) {
		sendBufsBytes[i].resize(sendBufsElements[i].size() * fieldByteSize);
		recBufBytes[i].resize(
				sendBufsElements[m_partyId].size() * fieldByteSize);
//        for(int j=0; j<sendBufsElements[i].size();j++) {
//            field->elementToBytes(sendBufsBytes[i].data() + (j * fieldByteSize), sendBufsElements[i][j]);
//        }

		field->elementVectorToByteVector(sendBufsElements[i], sendBufsBytes[i]);
	}

	//comm->roundfunctionI(sendBufsBytes, recBufBytes,7);
	roundFunctionSync(sendBufsBytes, recBufBytes, 7);

	int counter = 0;
	log4cpp::Category::getInstance(logcat).debug("%s: endnend.", __FUNCTION__);
	for (int k = M - numOfOutputGates; k < M; k++) {
		if (circuit.getGates()[k].gateType == OUTPUT
				&& circuit.getGates()[k].party == m_partyId) {
			for (int i = 0; i < N; i++) {

				x1[i] = field->bytesToElement(
						recBufBytes[i].data() + (counter * fieldByteSize));
			}

			// my output: reconstruct received shares
			if (!checkConsistency(x1, T)) {
				// someone cheated!
				log4cpp::Category::getInstance(logcat_out).warn(
						"%s: cheat detected.", __FUNCTION__);
				return;
			}
			log4cpp::Category::getInstance(logcat_out).notice(
					"%s: the result for %d is %s", __FUNCTION__,
					circuit.getGates()[k].input1,
					field->elementToString(interpolate(x1)).c_str());
			counter++;
		}
	}

	// close output file
	myfile.close();
}

template<class FieldType>
void ProtocolParty<FieldType>::roundFunctionSync(vector<vector<byte>> &sendBufs,
		vector<vector<byte>> &recBufs, int round) {

	recBufs[m_partyId] = move(sendBufs[m_partyId]);

	for (size_t pid = 0; pid < m_parties.size(); ++pid) {
		if (pid == m_partyId)
			continue;
		if (0 == m_cc->send(pid, sendBufs[pid].data(), sendBufs[pid].size()))
			log4cpp::Category::getInstance(logcat).debug(
					"%s: sent %lu bytes to party %lu", __FUNCTION__,
					sendBufs[pid].size(), pid);
		else {
			log4cpp::Category::getInstance(logcat).error("%s: cc send failed.",
					__FUNCTION__);
			exit(__LINE__);
		}
	}

	bool all_data_ready = true;
	do {
		process_network_events();
		for (size_t pid = 0; pid < m_parties.size(); ++pid) {
			if (pid == m_partyId)
				continue;
			all_data_ready = all_data_ready
					&& (m_parties[pid].data.size() >= recBufs[pid].size());
		}
	} while (!all_data_ready);

	for (size_t pid = 0; pid < m_parties.size(); ++pid) {
		if (pid == m_partyId)
			continue;
		log4cpp::Category::getInstance(logcat).debug(
				"%s: reading %lu bytes from party %lu", __FUNCTION__,
				recBufs[pid].size(), pid);
		memcpy(recBufs[pid].data(), m_parties[pid].data.data(),
				recBufs[pid].size());
		m_parties[pid].data.erase(m_parties[pid].data.begin(),
				m_parties[pid].data.begin() + recBufs[pid].size());
	}
}

template<class FieldType>
void ProtocolParty<FieldType>::roundFunctionSyncForP1(vector<byte> &myShare,
		vector<vector<byte>> &recBufs) {

	recBufs[m_partyId] = myShare;

	bool all_data_ready = true;
	do {
		process_network_events();
		for (size_t pid = 0; pid < m_parties.size(); ++pid) {
			if (pid == m_partyId)
				continue;
			all_data_ready = all_data_ready
					&& (m_parties[pid].data.size() >= recBufs[pid].size());
		}
	} while (!all_data_ready);

	for (size_t pid = 0; pid < m_parties.size(); ++pid) {
		if (pid == m_partyId)
			continue;
		memcpy(recBufs[pid].data(), m_parties[pid].data.data(),
				recBufs[pid].size());
		m_parties[pid].data.erase(m_parties[pid].data.begin(),
				m_parties[pid].data.begin() + recBufs[pid].size());
	}
}

template<class FieldType>
ProtocolParty<FieldType>::~ProtocolParty() {
	protocolTimer->writeToFile();
	delete protocolTimer;
	delete field;
	delete timer;
	//delete comm;

	int errcode = 0;
	if (0 != (errcode = pthread_cond_destroy(&m_comm_e))) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_cond_destroy() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
	}
	if (0 != (errcode = pthread_mutex_destroy(&m_elock))) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_mutex_destroy() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
	}
	if (0 != (errcode = pthread_mutex_destroy(&m_qlock))) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_mutex_destroy() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
	}
	delete m_cc;
}

template<class FieldType>
void ProtocolParty<FieldType>::push_comm_event(comm_evt * evt) {
	int errcode;
	if (0 != (errcode = pthread_mutex_lock(&m_qlock))) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_mutex_lock() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
		exit(__LINE__);
	}

	m_comm_q.push_back(evt);

	if (0 != (errcode = pthread_mutex_unlock(&m_qlock))) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_mutex_unlock() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
		exit(__LINE__);
	}

	if (0 != (errcode = pthread_mutex_lock(&m_elock))) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_mutex_lock() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
		exit(__LINE__);
	}

	if (0 != (errcode = pthread_cond_signal(&m_comm_e))) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_cond_signal() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
		exit(__LINE__);
	}

	if (0 != (errcode = pthread_mutex_unlock(&m_elock))) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_mutex_unlock() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
		exit(__LINE__);
	}
}

template<class FieldType>
void ProtocolParty<FieldType>::report_party_comm(const size_t party_id,
		const bool comm) {
	comm_conn_evt * pevt = new comm_conn_evt;
	pevt->type = comm_evt_conn;
	pevt->party_id = party_id;
	pevt->connected = comm;
	push_comm_event(pevt);
}

template<class FieldType>
void ProtocolParty<FieldType>::process_network_events() {
	int errcode;
	std::list<comm_evt *> comm_evts;
	if (0 != (errcode = pthread_mutex_lock(&m_elock))) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_mutex_lock() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
		exit(__LINE__);
	}

	struct timespec to;
	clock_gettime(CLOCK_REALTIME, &to);
	to.tv_sec += 1;
	if (0 != (errcode = pthread_cond_timedwait(&m_comm_e, &m_elock, &to))
			&& ETIMEDOUT != errcode) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_cond_timedwait() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
		exit(__LINE__);
	}

	if (0 != (errcode = pthread_mutex_unlock(&m_elock))) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_mutex_unlock() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
		exit(__LINE__);
	}

	if (0 != (errcode = pthread_mutex_lock(&m_qlock))) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_mutex_lock() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
		exit(__LINE__);
	}

	comm_evts.swap(m_comm_q);

	if (0 != (errcode = pthread_mutex_unlock(&m_qlock))) {
		char errmsg[256];
		log4cpp::Category::getInstance(logcat).error(
				"%s: pthread_mutex_unlock() failed with error %d : [%s].",
				__FUNCTION__, errcode, strerror_r(errcode, errmsg, 256));
		exit(__LINE__);
	}

	for (typename std::list<comm_evt *>::iterator i = comm_evts.begin();
			i != comm_evts.end(); ++i) {
		switch ((*i)->type) {
		case comm_evt_conn:
			m_parties[(*i)->party_id].conn = ((comm_conn_evt*) (*i))->connected;
			break;
		case comm_evt_msg:
			m_parties[(*i)->party_id].data.insert(
					m_parties[(*i)->party_id].data.end(),
					((comm_msg_evt*) (*i))->msg.begin(),
					((comm_msg_evt*) (*i))->msg.end());
			break;
		default:
			break;
		}
		delete (*i);
	}
	comm_evts.clear();
}

template<class FieldType>
void ProtocolParty<FieldType>::wait_for_peer_connections() {
	bool all_parties_connected = false;

	do {
		process_network_events();
		all_parties_connected = true;
		for (typename std::vector<peer_t>::iterator i = m_parties.begin();
				i != m_parties.end(); ++i) {
			if (i->id == m_partyId)
				continue;
			all_parties_connected = all_parties_connected && i->conn;
		}
	} while (!all_parties_connected);
}

template<class FieldType>
void ProtocolParty<FieldType>::on_comm_up_with_party(
		const unsigned int party_id) {
	report_party_comm(party_id, true);
}

template<class FieldType>
void ProtocolParty<FieldType>::on_comm_down_with_party(
		const unsigned int party_id) {
	report_party_comm(party_id, false);
}

template<class FieldType>
void ProtocolParty<FieldType>::on_comm_message(const unsigned int src_id,
		const unsigned char * msg, const size_t size) {
	comm_msg_evt * pevt = new comm_msg_evt;
	pevt->type = comm_evt_msg;
	pevt->party_id = src_id;
	pevt->msg.assign(msg, msg + size);
	push_comm_event(pevt);
}

#endif /* PROTOCOLPARTY_H_ */
