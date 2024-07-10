import cx from 'classnames';
import { FC, ReactNode, useCallback, useState } from 'react';
import {
  BsFillCheckCircleFill,
  BsFillExclamationOctagonFill,
  BsFillExclamationTriangleFill,
  BsFillInfoCircleFill,
} from 'react-icons/bs';

import config from '../../core/config';
import { useAppContext } from '../../core/context';
import { useTimeout } from '../../core/useTimeout';
import { dateToFromAgo } from '../../core/utils';
import { NotificationData, NotificationType } from '../../types';

export const Notifications: FC = () => {
  const {
    appContext: { notifications },
    setAppContext,
  } = useAppContext();

  // close a notifications == removing from the list in central state
  const close = useCallback(
    (id: number) =>
      setAppContext((state) => ({
        ...state,
        notifications: state.notifications.filter((n) => n.id !== id),
      })),
    [setAppContext],
  );
  console.log(notifications);
  return (
    <div
      className="toasts-container fixed-bottom"
      style={{ zIndex: 1056, left: 'auto', maxWidth: '50%' }}
    >
      {notifications.map((notification) => (
        <Notification
          key={notification.id}
          notification={notification}
          onClose={() => close(notification.id)}
        />
      ))}
    </div>
  );
};

const CLASSES_TOAST: Record<NotificationData['type'], string> = {
  success: '',
  info: '',
  warning: '',
  error: '',
};

const ICONS_TOAST: Record<NotificationData['type'], ReactNode> = {
  success: <BsFillCheckCircleFill className="text-success" />,
  info: <BsFillInfoCircleFill className="text-info" />,
  warning: <BsFillExclamationTriangleFill className="text-warning" />,
  error: <BsFillExclamationOctagonFill className="text-danger" />,
};

const Notification: FC<{
  notification: NotificationType;
  onClose?: () => void;
}> = ({ notification, onClose }) => {
  const [show, setShow] = useState<boolean>(true);

  const close = useCallback(() => {
    setShow(false);
    if (onClose) onClose();
  }, [setShow, onClose]);

  const { cancel, reschedule } = useTimeout(close, config.notificationTimeoutMs);

  return (
    <div
      className={cx('toast fade m-2', show ? 'show' : 'hide', CLASSES_TOAST[notification.type])}
      onMouseEnter={cancel}
      onMouseLeave={reschedule}
    >
      <div className="toast-header">
        {ICONS_TOAST[notification.type]}
        <strong className="ms-2 me-auto">{notification.title || notification.type}</strong>
        {notification.createdAt && <small>{dateToFromAgo(notification.createdAt)}</small>}

        <button type="button" className="btn-close" onClick={() => close()}></button>
      </div>

      <div className="toast-body">{notification.message}</div>
    </div>
  );
};

export default Notifications;
