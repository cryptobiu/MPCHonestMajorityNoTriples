//
// Created by meital on 19/03/17.
//
#ifndef VDMTranspose_H_
#define VDMTranspose_H_

#include <vector>
#include <stdio.h>
#include "TemplateField.h"

using namespace NTL;

template<typename FieldType>
class VDMTranspose {
private:
    int m_n,m_m;
    FieldType** m_matrix;
    TemplateField<FieldType> *field;
public:
    VDMTranspose(int n, int m, TemplateField<FieldType> *field);
    VDMTranspose() {};
    ~VDMTranspose();
    void InitVDMTranspose();
    void Print();
    void MatrixMult(std::vector<FieldType> &vector, std::vector<FieldType> &answer, int length);

    void allocate(int n, int m, TemplateField<FieldType> *field);
};


template<typename FieldType>
VDMTranspose<FieldType>::VDMTranspose(int n, int m, TemplateField<FieldType> *field) {
    this->m_m = m;
    this->m_n = n;
    this->field = field;
    this->m_matrix = new FieldType*[m_n];
    for (int i = 0; i < m_n; i++)
    {
        m_matrix[i] = new FieldType[m_m];
    }
}

template<typename FieldType>
void VDMTranspose<FieldType>::allocate(int n, int m, TemplateField<FieldType> *field) {

    this->m_m = m;
    this->m_n = n;
    this->field = field;
    this->m_matrix = new FieldType*[m_n];
    for (int i = 0; i < m_n; i++)
    {
        m_matrix[i] = new FieldType[m_m];
    }
}

template<typename FieldType>
void VDMTranspose<FieldType>::InitVDMTranspose() {
    vector<FieldType> alpha(m_m);
    for (int i = 0; i < m_m; i++) {
        alpha[i] = field->GetElement(i + 1);
    }

    for (int i = 0; i < m_m; i++) {
        m_matrix[0][i] = *(field->GetOne());
        for (int k = 1; k < m_n; k++) {
            m_matrix[k][i] = m_matrix[k-1][i] * (alpha[k]);
        }
    }
}

/**
 * the function print the matrix
 */
template<typename FieldType>
void VDMTranspose<FieldType>::Print()
{
    for (int i = 0; i < m_m; i++)
    {
        for(int j = 0; j < m_n; j++)
        {
            cout << (m_matrix[i][j]) << " ";

        }
        cout << " " << '\n';
    }

}

template<typename FieldType>
void VDMTranspose<FieldType>::MatrixMult(std::vector<FieldType> &vector, std::vector<FieldType> &answer, int length)
{
    for(int i = 0; i < length; i++)
    {
        // answer[i] = 0
        answer[i] = *(field->GetZero());

        for(int j=0; j < m_n; j++)
        {
            answer[i] += (m_matrix[i][j] * vector[j]);
        }
    }

}


//
template<typename FieldType>
VDMTranspose<FieldType>::~VDMTranspose() {
    for (int i = 0; i < m_n; i++) {
        delete[] m_matrix[i];
    }
    delete[] m_matrix;
}


#endif /* VDMTranspose_H_ */