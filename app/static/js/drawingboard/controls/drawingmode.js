DrawingBoard.Control.DrawingMode = DrawingBoard.Control.extend({

	name: 'drawingmode',

	defaults: {
		eraser: true,
		pencil: false,
		filler: false
	},

	initialize: function() {

		this.prevMode = this.board.getMode();

		$.each(["eraser", "pencil","filler"], $.proxy(function(k, value) {
			if (this.opts[value]) {
				if(value=="eraser"){ this.$el.append('<button id="'+value+'" class="drawing-board-control-drawingmode-' + value + '-button" data-mode="' + value + '" title="橡皮檫"></button>');}
				if(value=="pencil") { this.$el.append('<button id="'+value+'" class="drawing-board-control-drawingmode-' + value + '-button" data-mode="' + value + '" title="识别"></button>');}
			}
		}, this));

		this.$el.on('click', 'button[data-mode]', $.proxy(function(e) {
			var value = $(e.currentTarget).attr('data-mode');
			var mode = this.board.getMode();
			if (mode !== value) this.prevMode = mode;
			var newMode = mode === value ? this.prevMode : value;
			this.board.setMode( newMode );
			e.preventDefault();
		}, this));

		this.board.ev.bind('board:mode', $.proxy(function(mode) {
			this.toggleButtons(mode);
		}, this));

		this.toggleButtons( this.board.getMode() );
	},

	toggleButtons: function(mode) {
		this.$el.find('button[data-mode]').each(function(k, item) {
			var $item = $(item);
			$item.toggleClass('active', mode === $item.attr('data-mode'));
		});
	}

});
