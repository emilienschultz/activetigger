import { FC, useState } from 'react';
import { FaRegTrashAlt } from 'react-icons/fa';
import { useDeleteMessage, useGetMessages } from '../core/api';

export const ManageMessages: FC = () => {
  const [kind, setKind] = useState<string>('system');
  const { messages, reFetchMessages } = useGetMessages(kind, null);
  const { deleteMessage } = useDeleteMessage();

  const displayTime = (time: string) => {
    // 2025-12-11 10:50:57.786644 -> 2025-12-11 10:50
    return time.slice(0, time.indexOf(':') + 3);
  };
  return (
    <>
      <div className="horizontal">
        <div style={{ margin: '0px 20px' }}>Kind</div>
        <select onChange={(e) => setKind(e.target.value)} style={{ maxWidth: '300px' }}>
          <option value="system">System</option>
        </select>
      </div>
      <table id="message-table">
        <thead>
          <tr>
            <th scope="col" style={{ minWidth: '150px' }}>
              Time
            </th>
            <th scope="col" style={{ minWidth: '80px' }}>
              User
            </th>
            <th scope="col" style={{ minWidth: '80px' }}>
              Kind
            </th>
            <th scope="col">Message</th>
            <th scope="col" style={{ minWidth: '50px' }}>
              Delete
            </th>
          </tr>
        </thead>
        <tbody>
          {(messages || []).map((message) => (
            <tr key={message.id}>
              <td>{displayTime(message.time)}</td>
              <td>{message.created_by}</td>
              <td>{message.kind}</td>
              <td>{message.content}</td>
              <td>
                <div
                  title="Delete"
                  className="cursor-pointer trash-wrapper"
                  onClick={() => {
                    deleteMessage(message.id);
                    reFetchMessages();
                  }}
                >
                  <FaRegTrashAlt />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
};
