import { FC, useState } from 'react';
import { FaRegTrashAlt } from 'react-icons/fa';
import { useDeleteMessage, useGetMessages } from '../core/api';

export const ManageMessages: FC = () => {
  const [kind, setKind] = useState<string>('system');
  const { messages, reFetchMessages } = useGetMessages(kind, null);
  const { deleteMessage } = useDeleteMessage();

  return (
    <div className="container">
      <div className="row">
        <label className="d-flex align-items-center">
          Kind
          <select onChange={(e) => setKind(e.target.value)} className="form-select w-25 mx-3">
            <option value="system">System</option>
          </select>
        </label>
        <table className="table table-hover">
          <thead>
            <tr>
              <th scope="col">Time</th>
              <th scope="col">User</th>
              <th scope="col">Kind</th>
              <th scope="col">Message</th>
              <th scope="col">Delete</th>
            </tr>
          </thead>
          <tbody>
            {(messages || []).map((message) => (
              <tr key={message.id}>
                <td>{message.time}</td>
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
      </div>
    </div>
  );
};
