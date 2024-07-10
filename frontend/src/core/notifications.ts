import { useCallback } from 'react';

import { NotificationData } from '../types';
import { useAppContext } from './context';

let INCREMENTAL_ID = 1;
export function useNotifications() {
  const { setAppContext } = useAppContext();

  const notify = useCallback(
    (notif: NotificationData) => {
      const id = ++INCREMENTAL_ID;
      setAppContext((state) => ({
        ...state,
        notifications: [{ id, createdAt: new Date(), ...notif }, ...state.notifications],
      }));
    },
    [setAppContext],
  );

  return { notify };
}
