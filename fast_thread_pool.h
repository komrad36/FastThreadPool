/*******************************************************************
*
*    Author: Kareem Omar
*    kareem.h.omar@gmail.com
*    https://github.com/komrad36
*
*    Last updated Oct 18, 2021
*******************************************************************/

#pragma once

#include <atomic>
#include <cstdint>
#ifdef __AVX__
#include <immintrin.h>
#endif

class FastThreadPool
{
private:
    struct FastThreadPoolItem
    {
        FastThreadPoolItem* m_next;
        void (*m_f)();
    };

    static void Pause()
    {
#ifdef __AVX__
        _mm_pause();
#endif
    }

public:
    FastThreadPool()
    {
        // S1
        m_pHead.store(nullptr, std::memory_order_relaxed);

        // S2
        // HAPPENS-AFTER: S1
        // SYNC-WITH: THREAD-LAUNCH
        m_pTail.store(nullptr, std::memory_order_release);
    }

    void AddWork(void f())
    {
        FastThreadPoolItem* pItem = new FastThreadPoolItem;

        // S3
        pItem->m_next = nullptr;
        pItem->m_f = f;

        FastThreadPoolItem* pTail = m_pTail.load(std::memory_order_relaxed);

        do
        {
            while ((uintptr_t)pTail & 1)
            {
                pTail = m_pTail.load(std::memory_order_relaxed);
                Pause();
            }

            // L1
            //
            // S4
            // HAPPENS-AFTER: S3
            // SYNC-WITH: L3
        } while (!m_pTail.compare_exchange_weak(pTail, pTail ? (FastThreadPoolItem*)((uintptr_t)pTail | 1) : pItem, std::memory_order_release, std::memory_order_relaxed));

        if (pTail)
        {
            // S5
            pTail->m_next = pItem;

            // S6
            // HAPPENS-AFTER: S5
            // SYNC-WITH: L3
            m_pTail.store(pItem, std::memory_order_release);
        }
        else
        {
            FastThreadPoolItem* pHead = m_pHead.load(std::memory_order_relaxed);

            do
            {
                while (pHead)
                {
                    pHead = m_pHead.load(std::memory_order_relaxed);
                    Pause();
                }

                // L2
                //
                // S7
                // HAPPENS-AFTER: S3, S4
                // SYNC-WITH: L3
            } while (!m_pHead.compare_exchange_weak(pHead, pItem, std::memory_order_release, std::memory_order_relaxed));
        }
    }

    void (*RemoveWork(void))()
    {
        // L3
        // HAPPENS-BEFORE: L4, L5
        FastThreadPoolItem* pHead = m_pHead.load(std::memory_order_relaxed);

        do
        {
            while (pHead == (FastThreadPoolItem*)(1))
            {
                pHead = m_pHead.load(std::memory_order_relaxed);
                Pause();
            }

            if (!pHead)
            {
                break;
            }

            // also L3
            // HAPPENS-BEFORE: L4, L5
            //
            // S8
        } while (!m_pHead.compare_exchange_weak(pHead, (FastThreadPoolItem*)(1), std::memory_order_acquire, std::memory_order_acquire));

        if (pHead)
        {
            void (*f)() = pHead->m_f;

            // L4
            FastThreadPoolItem* pTail = m_pTail.load(std::memory_order_relaxed);

            do
            {
                while ((uintptr_t)pTail == ((uintptr_t)pHead | 1))
                {
                    pTail = m_pTail.load(std::memory_order_relaxed);
                    Pause();
                }

                if (pTail != pHead)
                {
                    break;
                }

                // also L4
                // S9
            } while (!m_pTail.compare_exchange_weak(pTail, nullptr, std::memory_order_relaxed, std::memory_order_relaxed));

            // L5
            FastThreadPoolItem* pNext = pTail == pHead ? nullptr : pHead->m_next;

            // S10
            m_pHead.store(pNext, std::memory_order_relaxed);

            delete pHead;

            return f;
        }

        return nullptr;
    }

    ~FastThreadPool()
    {
        FastThreadPoolItem* pItem = m_pHead.load(std::memory_order_relaxed);
        while (pItem)
        {
            FastThreadPoolItem* pPrev = pItem;
            pItem = pItem->m_next;
            delete pPrev;
        }
    }

private:
    alignas(64) std::atomic<FastThreadPoolItem*> m_pHead;
    alignas(64) std::atomic<FastThreadPoolItem*> m_pTail;
};
