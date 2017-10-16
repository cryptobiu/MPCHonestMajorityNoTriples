
#include <stdlib.h>
#include "Protocol.h"
#include "ZpMersenneIntElement.h"
#include "ZpMersenneLongElement.h"
#include "ZpKaratsubaElement.h"
#include <smmintrin.h>
#include <inttypes.h>
#include <stdio.h>
#include <x86intrin.h>



__m128 _mm_mod_ps2(const __m128& a, const __m128& aDiv){
    __m128 c = _mm_div_ps(a,aDiv);
    __m128i i = _mm_cvttps_epi32(c);
    __m128 cTrunc = _mm_cvtepi32_ps(i);
    __m128 base = _mm_mul_ps(cTrunc, aDiv);
    __m128 r = _mm_sub_ps(a, base);
    return r;
}


void mul128(__m128i a, __m128i b, __m128i *res1, __m128i *res2)
{
    __m128i tmp3, tmp4, tmp5, tmp6;

    tmp3 = _mm_clmulepi64_si128(a, b, 0x00);
    tmp4 = _mm_clmulepi64_si128(a, b, 0x10);
    tmp5 = _mm_clmulepi64_si128(a, b, 0x01);
    tmp6 = _mm_clmulepi64_si128(a, b, 0x11);

    tmp4 = _mm_xor_si128(tmp4, tmp5);
    tmp5 = _mm_slli_si128(tmp4, 8);
    tmp4 = _mm_srli_si128(tmp4, 8);
    tmp3 = _mm_xor_si128(tmp3, tmp5);
    tmp6 = _mm_xor_si128(tmp6, tmp4);
    // initial mul now in tmp3, tmp6
    *res1 = tmp3;
    *res2 = tmp6;
}


void multUlong(unsigned long a, unsigned long b, unsigned long *res1, unsigned long *res2){

    unsigned int alow = ((int*)&a)[0];
    unsigned int ahi = ((int*)&a)[1];
    unsigned int blow = ((int*)&b)[0];
    unsigned int bhi = ((int*)&b)[1];

    uint64_t    a_lo = (uint32_t)a;
    uint64_t    a_hi = a >> 32;
    uint64_t    b_lo = (uint32_t)b;
    uint64_t    b_hi = b >> 32;

    uint64_t a_x_b_hi =  ahi * b_hi;
    uint64_t a_x_b_mid = a_hi * b_lo;
    uint64_t b_x_a_mid = b_hi * a_lo;
    uint64_t a_x_b_lo =  a_lo * b_lo;


    *res1 = a_x_b_lo + ((uint64_t)(uint32_t)a_x_b_mid +
                       (uint64_t)(uint32_t)b_x_a_mid)<<32;


    unsigned long carry_bit = ((uint64_t)(uint32_t)a_x_b_mid +
                 (uint64_t)(uint32_t)b_x_a_mid +
                 (a_x_b_lo >> 32) ) >> 32;

    *res2 = a_x_b_hi +
                         (a_x_b_mid >> 32) + (b_x_a_mid >> 32) +
                         carry_bit;


}

unsigned long mersenneAdd(unsigned long high, unsigned long low){

    unsigned long low61 = (low & 2305843009213693951);
    unsigned long low61to64 = (low>>61);
    unsigned long highShift3 = (high<<3);

    unsigned long res = low61 + low61to64 + highShift3;

    if(res >= 2305843009213693951)
        res-= 2305843009213693951;

    return res;


}


void multkarm(__m128i *c1, __m128i *c0, __m128i b,
              __m128i a)
{
    __m128i t1, t2;
    *c0 = _mm_clmulepi64_si128(a, b, 0x00);
    *c1 = _mm_clmulepi64_si128(a, b, 0x11);
    t1 = _mm_shuffle_epi32(a, 0xEE);
    t1 = _mm_xor_si128(a, t1);
    t2 = _mm_shuffle_epi32(b, 0xEE);
    t2 = _mm_xor_si128(b, t2);
    t1 = _mm_clmulepi64_si128(t1, t2, 0x00);
    t1 = _mm_xor_si128(*c0, t1);
    t1 = _mm_xor_si128(*c1, t1);
    t2 = t1;
    t1 = _mm_slli_si128(t1, 8);
    t2 = _mm_srli_si128(t2, 8);
    *c0 = _mm_xor_si128(*c0, t1);
    *c1 = _mm_xor_si128(*c1, t2);

}



/**
 * The main structure of our protocol is as follows:
 * 1. Initialization Phase: Initialize some global variables (parties, field, circuit, etc).
 * 2. Preparation Phase: Prepare enough random double-sharings: a random double-sharing is a pair of
 *  two sharings of the same random value, one with degree t, and one with degree 2t. One such double-
 *  sharing is consumed for multiplying two values. We also consume double-sharings for input gates
 *  and for random gates (this is slightly wasteful, but we assume that the number of multiplication
 *  gates is dominating the number of input and random gates).
 * 3. Input Phase: For each input gate, reconstruct one of the random sharings towards the input party.
 *  Then, all input parties broadcast a vector of correction values, namely the differences of the inputs
 *  they actually choose and the random values they got. These correction values are then added on
 *  the random sharings.
 * 4. Computation Phase: Walk through the circuit, and evaluate as many gates as possible in parallel.
 *  Addition gates and random gates can be evaluated locally (random gates consume a random double-
 *  sharing). Multiplication gates are more involved: First, every party computes local product of the
 *  respective shares; these shares form de facto a 2t-sharing of the product. Then, from this sharing,
 *  a degree-2t sharing of a random value is subtracted, the difference is reconstructed and added on
 *  the degree-t sharing of the same random value.
 * 5. Output Phase: The value of each output gate is reconstructed towards the corresponding party.
 * @param argc
 * @param argv[1] = id of parties (1,...,N)
 * @param argv[2] = N: number of parties
 * @param argv[3] = path of inputs file
 * @param argv[4] = path of output file
 * @param argv[5] = path of circuit file
 * @param argv[6] = address
 * @param argv[7] = fieldType
 * @return
 */


int main(int argc, char* argv[])
{

//    int elem = 1000;
//
//    __m128i left =  _mm_set_epi32(0,0, 7, 7);
//    __m128i right =  _mm_set_epi32(0,0, 7, 7);
//    __m128i result = _mm_clmulepi64_si128(left, right, 0);
//
//    __m128i res1, res2;
//   // __m128i kar;
//
//
//    mul128(left,right,&res1, &res2);
//    //multkarm(&res2,&kar,left,right);
//
//
//
//    uint64_t a = 100082619497;
//    uint64_t b = 100082619497;
//    uint64_t  d;
//    long long unsigned int c;
//    d = _mulx_u64(a, b, &c);
//    cout<<"mult128 : "<<(unsigned long)c<<","<<(unsigned long)d<<endl;
//
//    //d = a*b;
//
//    cout<<"mult64 : "<<(unsigned long)d<<endl;
//
//    unsigned long resUlong;
//    unsigned long res2Ulong;
//
//    multUlong(a,b,&resUlong, &res2Ulong);
//
//    cout<<"multUlong result : "<<(unsigned long)res2Ulong<<","<<(unsigned long)resUlong<<endl;
//
//    unsigned long mer = mersenneAdd(c, d);
//
//    cout<<"Mersenne result is : "<<mer<<endl;
//
//
//
//    mpz_t rop;
//    mpz_t op1;
//    mpz_t op2;
//    mpz_t resultgmp;
//    mpz_t dgmp;
//
//    mpz_init_set_str (op1, "100082619497", 10);
//    mpz_init_set_str (op2, "100082619497", 10);
//    mpz_init_set_str (dgmp, "2305843009213693951", 10);
//
//    mpz_init(rop);
//    mpz_init(resultgmp);
//
//    mpz_mul (rop, op1, op2);
//    mpz_mod (resultgmp, rop, dgmp);
//
//
//    cout << "result of a*b is : " << resultgmp << endl;
//
//
//
//    unsigned long x = 2147483647;
//    unsigned long y = 2147483647;
//
//    long resLong = x*y;
//
//    cout<<"mult64Long : "<<resLong<<endl;
//
//    //_mm_extract_epi64(left,0);
//    //cout<< "left is " <<_mm_extract_epi64(left,0) <<endl;
//    cout<< "res1 is " <<((unsigned long*)&res1)[0] << " , "<<((unsigned long*)&res1)[1];
//    cout<< "result is " <<((unsigned long*)&result)[0] << " , "<<((unsigned long*)&result)[1];
//    //cout<< "kar is " <<((unsigned long*)&kar)[0] << " , "<<((unsigned long*)&kar)[1];
//
//
//    //generate a pseudo random generator to generate the keys
//    PrgFromOpenSSLAES prg(100*1000000);
//    auto randomKey = prg.generateKey(128);
//    prg.setKey(randomKey);
//
//
//    vector<ZpKaratsubaElement> kar(20000000);
//
//
//
//    for(int i=0; i<20000000; i++){
//
//        ZpKaratsubaElement a(prg.getRandom64());
//        kar[i] = a;
//    }
//
//
//
//    auto duration_avg = 0;
//
//
//
//
////    mpz_t rop;
////    mpz_t op1;
////    mpz_t op2;
////    mpz_t result;
////    mpz_t d;
////
////    mpz_init_set_str (op1, "181254622435", 10);
////    mpz_init_set_str (op2, "850793430687", 10);
////    mpz_init_set_str (d, "1071482619497", 10);
////
////    mpz_init(rop);
////    mpz_init(result);
////
////    mpz_mul (rop, op1, op2);
////    mpz_mod (result, rop, d);
////
////    cout << "result of b*a is : " << result << endl;
////
////    mpz_mul (rop, op2, op1);
////    mpz_mod (result, rop, d);
////
////
////    cout << "result of a*b is : " << result << endl;
////
////
////
////
////    duration_avg = 0;
////    auto t1 = high_resolution_clock::now();
////
////    for(int i=0; i<10000000; i++) {
////
////
////
////            mpz_mul (rop, op1, op2);
////            mpz_mod (result, rop, d);
////
////
////
////    }
////    auto t2 = high_resolution_clock::now();
////
////    auto duration = duration_cast<microseconds>(t2 - t1).count();
////
////    duration_avg += duration;
////    //duration_avg = duration_avg;
////
////
////    cout << "time in milliseconds : " << duration_avg << endl;
//
//    duration_avg = 0;
//    ZpKaratsubaElement aKar(181254622435);
//    ZpKaratsubaElement bKar(850793430687);
//
//    auto ab = aKar*bKar;
//    auto ba = bKar*aKar;
//
//    cout<<"a*b = " <<ab.elem<<endl;
//    cout<<"b*a = " <<ba.elem<<endl;
//
//
//    ZpKaratsubaElement cKar;
//    ZpKaratsubaElement small(500);
//
//
//    //ZpMersenneIntElement p(1071482619497);
//
//    auto t1 = high_resolution_clock::now();
//    for(int i=0; i<10000000; i++) {
//
//        //c = a*b;
//            cKar = kar[i]*kar[2*i];
//            //c = small*small;//kar[2*i];
//
//    }
//     auto t2 = high_resolution_clock::now();
//
//     auto duration = duration_cast<microseconds>(t2 - t1).count();
//
//    duration_avg += duration;
//    //duration_avg = duration_avg;
//
//
//    cout << "time in milliseconds : " << duration_avg << endl;
//
//
//    mpz_t ropMen;
//    mpz_t op1Men;
//    mpz_t op2Men;
//    mpz_t resultMen;
//    mpz_t dMen;
//
//    mpz_init_set_str (op1Men, "23058430092136939", 10);
//    mpz_init_set_str (op2Men, "23058430092136939", 10);
//    mpz_init_set_str (dMen, "2305843009213693951", 10);
//
//    mpz_init(ropMen);
//    mpz_init(resultMen);
//
//    mpz_mul (ropMen, op1Men, op2Men);
//    mpz_mod (resultMen, ropMen, dMen);
//
//    cout << "result of b*a is : " << resultMen << endl;
//
//    mpz_mul (ropMen, op2Men, op1Men);
//    mpz_mod (resultMen, ropMen, dMen);
//
//
//
//
//
//    duration_avg = 0;
//    t1 = high_resolution_clock::now();
//
//    for(int i=0; i<10000000; i++) {
//
//
//
//        mpz_mul (ropMen, op1Men, op2Men);
//        mpz_mod (resultMen, ropMen, dMen);
//
//
//
//    }
//    t2 = high_resolution_clock::now();
//
//    duration = duration_cast<microseconds>(t2 - t1).count();
//
//    duration_avg += duration;
//
//    cout << "time in micro for mersenne gmp : " << duration_avg << endl;
//
//    int timesfield = 1000000;
//    ZZ_p::init(ZZ(2147483647));
//
//    //testing the mersenne field
//
//    ZpMersenneLongElement aMerLong(100082619497);
//    ZpMersenneLongElement bMerLong(100082619497);
//
//    ZpMersenneLongElement cMerLong(2147483647);
//
//    c = a+b;
//
//    ZpMersenneLongElement multLong;
//    ZpMersenneLongElement divLong(2147483647);
//
//
//
//    t1 = high_resolution_clock::now();
//    for(int i=0; i<timesfield; i++){
//        multLong = aMerLong*bMerLong;
//    }
//    t2 = high_resolution_clock::now();
//
//    cout<<"result for mersenne long implemented field is:"<<multLong<<endl;
//
//    duration = duration_cast<microseconds>(t2-t1).count();
//    cout << "time in milliseconds for Mersenne long" << timesfield<< " mults: " << duration << endl;
//
//
//
//    //testing the mersenne field
//
//    ZpMersenneIntElement aMer(2147483646);
//    ZpMersenneIntElement bMer(2147483643);
//
//    ZpMersenneIntElement cMer(2147483647);
//
//    c = a+b;
//
//    ZpMersenneIntElement mult;
//    ZpMersenneIntElement div(2147483647);
//
//
//
//    t1 = high_resolution_clock::now();
//    for(int i=0; i<timesfield; i++){
//        mult = aMer*bMer;
//    }
//    t2 = high_resolution_clock::now();
//
//    duration = duration_cast<microseconds>(t2-t1).count();
//    cout << "time in milliseconds for Mersenne int " << timesfield<< " mults: " << duration << endl;
//
//
//
//
//
//
//    return 0;

/*
    t1 = high_resolution_clock::now();
    for(int i=0; i<timesfield; i++){
        div = a/b;
    }
    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    cout << "time in milliseconds for " << timesfield << " divs: " << duration << endl;


    cout<<"a + b = " << c.elem <<endl;
    cout<<"a * b = " << mult.elem<<endl;
    cout<<"a / b = " << div.elem<<endl;
    cout<<"a - b = " << a-b<<endl;


    cout<< "-1 % 2147483647 is " << (-1 % 2147483647) << endl;


    ZZ_p::init(ZZ(2147483647));

    ZZ_p x(2147483640);
    ZZ_p y(2147483641);
    ZZ_p divZ(5);
    ZZ_p multZ(5);


    t1 = high_resolution_clock::now();
    for(int i=0; i<timesfield; i++){
        multZ = x*y;
    }
    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    cout << "time in milliseconds for " << timesfield<< " mults: " << duration << endl;




    t1 = high_resolution_clock::now();
    for(int i=0; i<timesfield; i++){
        divZ = x/y;
    }
    t2 = high_resolution_clock::now();

    duration = duration_cast<milliseconds>(t2-t1).count();
    cout << "time in milliseconds for " << timesfield << " divs: " << duration << endl;



    cout<<"zzp : a / b = " << to_uint(rep(divZ))<<endl;

    cout<<"zzp: a * b = " << (x*y) <<endl;

    cout<<"zzp: a - b = " << (x-y) <<endl;

    return 0;
*/
    if(argc != 11)
    {
        cout << "wrong number of arguments";
        return 0;
    }

    int times = 5;

    //string outputTimerFileName = string(argv[5]) + "Times" + string(argv[1]) + ".csv";
    string outputTimerFileName = string(argv[5]) + "Times" + string(argv[1]) + argv[6] + argv[7] + argv[8] + argv[9] + ".csv";
    ProtocolTimer p(times, outputTimerFileName);

    string fieldType(argv[6]);



    if(fieldType.compare("ZpMensenne") == 0)
    {
        TemplateField<ZpMersenneIntElement> *field = new TemplateField<ZpMersenneIntElement>(2147483647);

        Protocol<ZpMersenneIntElement> protocol(atoi(argv[2]), atoi(argv[1]), field, argv[3], argv[4], argv[5], &p, argv[7], argv[8], argv[9],atoi(argv[10]));
        auto t1 = high_resolution_clock::now();
        for(int i=0; i<times; i++) {
            protocol.run(i);
        }
        auto t2 = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(t2-t1).count();
        cout << "time in milliseconds for " << times << " runs: " << duration << endl;

        delete field;

        p.writeToFile();

        cout << "end main" << '\n';

    }
    else if(fieldType.compare("ZpMensenne61") == 0)
    {




        TemplateField<ZpMersenneLongElement> *field = new TemplateField<ZpMersenneLongElement>(0);

        Protocol<ZpMersenneLongElement> protocol(atoi(argv[2]), atoi(argv[1]), field, argv[3], argv[4], argv[5], &p, argv[7], argv[8], argv[9],atoi(argv[10]));
        auto t1 = high_resolution_clock::now();
        for(int i=0; i<times; i++) {
            protocol.run(i);
        }
        auto t2 = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(t2-t1).count();
        cout << "time in milliseconds for " << times << " runs: " << duration << endl;

        delete field;

        p.writeToFile();

        cout << "end main" << '\n';





        //        mpz_t rop;
//    mpz_t op1;
//    mpz_t op2;
//    mpz_t resultgmp;
//    mpz_t dgmp;
//
//    mpz_init_set_str (op1, "2305843009213693948", 10);
//    mpz_init_set_str (op2, "2305843009213693950", 10);
//    mpz_init_set_str (dgmp, "2305843009213693951", 10);
//
//    mpz_init(rop);
//    mpz_init(resultgmp);
//
//    mpz_mul (rop, op1, op2);
//    mpz_mod (resultgmp, rop, dgmp);
//
//
//    cout << "result of a*b is : " << resultgmp << endl;
////
////    mpz_t d;
////    mpz_t result;
////    mpz_t mpz_elem;
////    mpz_t mpz_me;
////    mpz_init_set_str (d, "2305843009213693951", 10);
////    mpz_init(mpz_elem);
////    mpz_init(mpz_me);
////
////    mpz_set_ui(mpz_elem, f2.elem);
////    mpz_set_ui(mpz_me, elem);
////
////    mpz_init(result);
////
////    mpz_invert ( result, mpz_elem, d );
////
////    mpz_mul (result, result, mpz_me);
////    mpz_mod (result, result, d);
//
//
//   // unsigned long res = mpz_get_ui(result);
//
//    ZpMersenneLongElement aMerLong(2305843009213693948);
//    ZpMersenneLongElement bMerLong(2305843009213693950);
//
//
//    ZpMersenneLongElement multLong;
//    ZpMersenneLongElement divLong;
//    ZpMersenneLongElement subLong;
//        ZpMersenneLongElement addLong;
//
//        multLong = aMerLong*bMerLong;
//        divLong = aMerLong/bMerLong;
//        subLong = aMerLong - bMerLong;
//        addLong = aMerLong + bMerLong;
//
//        cout<<"multLong : " << multLong<<endl;
//        cout<<"divLong : " << divLong<<endl;
//        cout<<"subLong : " << subLong<<endl;
//        cout<<"addLong : " << addLong<<endl;
//
//
//
//
//


    }

    else if(fieldType.compare("ZpKaratsuba") == 0) {
        TemplateField<ZpKaratsubaElement> *field = new TemplateField<ZpKaratsubaElement>(0);


        Protocol<ZpKaratsubaElement> protocol(atoi(argv[2]), atoi(argv[1]), field, argv[3], argv[4], argv[5], &p,
                                              argv[7], argv[8], argv[9],atoi(argv[10]));
        auto t1 = high_resolution_clock::now();
        for (int i = 0; i < times; i++) {
            protocol.run(i);
        }
        auto t2 = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(t2 - t1).count();
        cout << "time in milliseconds for " << times << " runs: " << duration << endl;

        delete field;

        p.writeToFile();

        cout << "end main" << '\n';
    }



    else if(fieldType.compare("GF2m") == 0)
    {
        TemplateField<GF2E> *field = new TemplateField<GF2E>(8);

        Protocol<GF2E> protocol(atoi(argv[2]), atoi(argv[1]), field, argv[3], argv[4], argv[5], &p, argv[7], argv[8], argv[9],atoi(argv[10]));
        auto t1 = high_resolution_clock::now();
        for(int i=0; i<times; i++) {
            protocol.run(i);
        }
        auto t2 = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(t2-t1).count();
        cout << "time in milliseconds for " << times << " runs: " << duration << endl;

        delete field;

        p.writeToFile();

        cout << "end main" << '\n';
    }

    else if(fieldType.compare("Zp") == 0)
    {
        TemplateField<ZZ_p> * field = new TemplateField<ZZ_p>(2147483647);

        Protocol<ZZ_p> protocol(atoi(argv[2]), atoi(argv[1]),field, argv[3], argv[4], argv[5], &p, argv[7], argv[8], argv[9],atoi(argv[10]));

        auto t1 = high_resolution_clock::now();
        for(int i=0; i<times; i++) {
            protocol.run(i);
        }
        auto t2 = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(t2-t1).count();
        cout << "time in milliseconds for " << times << " runs: " << duration << endl;

        delete field;

        p.writeToFile();

        cout << "end main" << '\n';

    }

    return 0;
}
