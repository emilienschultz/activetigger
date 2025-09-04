import { FC } from 'react';
import { useForm } from 'react-hook-form';
import { useSendMessage } from '../../core/api';

type NewMessage = {
  content: string;
  kind: string;
  to_user?: string;
  to_project?: string;
};

export const SendMessage: FC = () => {
  const { sendMessage } = useSendMessage();

  const { handleSubmit, register, reset } = useForm<NewMessage>({
    defaultValues: {
      kind: 'system',
      content: '',
    },
  });

  const postForm = async function (data: NewMessage) {
    sendMessage(data.content, data.kind);
    reset();
  };

  return (
    <div className="container">
      <div className="row">
        <form onSubmit={handleSubmit(postForm)}>
          <label className="form-label">
            Kind
            <select {...register('kind')} className="form-select w-25">
              <option value="system">System</option>
            </select>
          </label>
          <label className="form-label">
            Message
            <textarea className="form-control" placeholder="Message" {...register('content')} />
          </label>
          <button type="submit" className="btn btn-primary w-100">
            Send
          </button>
        </form>
      </div>
    </div>
  );
};
