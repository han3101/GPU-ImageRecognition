#pragma once
#include <vector>
#include <stdexcept>

class DisjointSet {
public:
    
    DisjointSet(int length) {
        if (length == 0) {
            throw std::invalid_argument("DisjointSet length not specified.");
        }
        this->m_length = length;
        m_parent.resize(length);
        for (int i = 0; i < length; ++i) {
            m_parent[i] = i;
        }
    }

    /**
     * @brief Finds a pointer to the representative of the set containing i.
     * 
     * @param i The element whose set representative is to be found.
     * @return size_t The representative set of i.
     */

    // Recursion causes path compression
    int find(int i) {
        if (m_parent[i] == i) {
            return i;
        } else {
            return m_parent[i] = find(m_parent[i]);
        }
    }

    /**
     * @brief Unites two dynamic sets containing objects i and j, say Si and Sj, into
     * a new set that Si ∪ Sj, assuming that Si ∩ Sj = ∅;
     * 
     * @param i An element in the first set.
     * @param j An element in the second set.
     */
    void unite(int i, int j) {
        int iRepresentative = find(i);
        int jRepresentative = find(j);
        m_parent[iRepresentative] = jRepresentative;
    }

    int size() {
        return m_length;
    }

private:
    int m_length;
    std::vector<int> m_parent;
};